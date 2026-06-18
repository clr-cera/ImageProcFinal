"""Classical HOG + linear-SVM sliding-window cup detector (no deep learning).

Pipeline:
  1. Load images from images_ready/ and their YOLO-format boxes from labels/.
  2. Build positive samples (cup crops + horizontal flips) and random negatives.
  3. Describe each window with a HOG feature vector and train a linear SVM.
  4. One round of hard-negative mining to suppress false positives.
  5. Detect via an image pyramid + sliding window + non-maximum suppression.
  6. Report precision/recall/F1 on a held-out split and save annotated images
     to detections/ (green = ground truth, red = detection).
"""

import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, io
from skimage.feature import hog
from skimage.transform import resize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

IMG_DIR = Path("images_ready")
LABEL_DIR = Path("labels/labels")
OUT_DIR = Path("detections")

WINDOW = (64, 64)  # (height, width) of the HOG detection window
HOG_KW = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    transform_sqrt=True,
    feature_vector=True,
    channel_axis=None,
)

NEG_PER_IMAGE = 20  # random negative windows sampled per training image
PYRAMID_DOWNSCALE = 1.2  # image-pyramid shrink factor between levels
STEP = 12  # sliding-window stride in pixels
NMS_IOU = 0.3  # overlap above which detections are merged
MATCH_IOU = 0.5  # overlap required to count a detection as correct
SWEEP_MIN = -1.0  # collect all windows scoring above this, then sweep cutoffs
THRESHOLDS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0]  # operating points to report
VIS_THRESHOLD = 0.0  # cutoff used for the saved annotated images
SEED = 0

RNG = np.random.default_rng(SEED)


def load_gray(path):
    """Load an image as a float grayscale array in [0, 1]."""
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    return img.astype(np.float64)


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_dataset():
    """Return [(image_path, gray_array, [boxes_xyxy_pixels]), ...]."""
    samples = []
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        stem = re.sub(r"^[0-9a-f]+-", "", label_path.stem)  # <hash>-image_N -> image_N
        img_path = IMG_DIR / f"{stem}.png"
        if not img_path.exists():
            print(f"skip {label_path.name}: no matching image")
            continue
        gray = load_gray(img_path)
        h, w = gray.shape
        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            boxes.append(clip_box(box, w, h))
        samples.append((img_path, gray, boxes))
    return samples


def window_feature(patch):
    """HOG descriptor of an arbitrary crop, resized to the detection window."""
    patch = resize(patch, WINDOW, anti_aliasing=True)
    return hog(patch, **HOG_KW)


def positives(samples):
    feats = []
    for _, gray, boxes in samples:
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            patch = gray[y1:y2, x1:x2]
            feats.append(window_feature(patch))
            feats.append(window_feature(patch[:, ::-1]))  # horizontal flip
    return feats


def negatives(samples, n_per_image=NEG_PER_IMAGE):
    feats = []
    for _, gray, boxes in samples:
        h, w = gray.shape
        count = attempts = 0
        while count < n_per_image and attempts < n_per_image * 20:
            attempts += 1
            size = int(RNG.integers(32, min(h, w)))
            x1 = int(RNG.integers(0, w - size + 1))
            y1 = int(RNG.integers(0, h - size + 1))
            cand = [x1, y1, x1 + size, y1 + size]
            if all(iou(cand, b) < 0.1 for b in boxes):
                feats.append(window_feature(gray[y1 : y1 + size, x1 : x1 + size]))
                count += 1
    return feats


def train_svm(pos, neg):
    X = np.array(pos + neg)
    y = np.array([1] * len(pos) + [0] * len(neg))
    # class_weight balances the heavy negative:positive ratio (esp. after
    # hard-negative mining), otherwise the SVM rarely predicts "cup".
    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=20000, dual="auto", class_weight="balanced"),
    )
    clf.fit(X, y)
    return clf


def detect(clf, gray, threshold=VIS_THRESHOLD):
    """Pyramid + sliding-window detection. Returns [(score, box_xyxy), ...]."""
    h0, w0 = gray.shape
    win_h, win_w = WINDOW
    dets = []
    scale = 1.0
    while True:
        img = gray if scale == 1.0 else resize(
            gray, (max(1, int(h0 * scale)), max(1, int(w0 * scale))), anti_aliasing=True
        )
        sh, sw = img.shape
        if sh < win_h or sw < win_w:
            break
        feats, coords = [], []
        for y in range(0, sh - win_h + 1, STEP):
            for x in range(0, sw - win_w + 1, STEP):
                feats.append(hog(img[y : y + win_h, x : x + win_w], **HOG_KW))
                coords.append((x, y))
        if feats:
            scores = clf.decision_function(np.array(feats))
            inv = 1.0 / scale
            for (x, y), s in zip(coords, scores):
                if s > threshold:
                    box = [x * inv, y * inv, (x + win_w) * inv, (y + win_h) * inv]
                    dets.append((float(s), box))
        scale /= PYRAMID_DOWNSCALE
    return dets


def nms(dets, iou_thr=NMS_IOU):
    keep = []
    for score, box in sorted(dets, key=lambda d: d[0], reverse=True):
        if all(iou(box, kb) < iou_thr for _, kb in keep):
            keep.append((score, box))
    return keep


def hard_negatives(clf, samples):
    """High-scoring windows that do not overlap any cup become extra negatives."""
    feats = []
    for _, gray, boxes in samples:
        h, w = gray.shape
        for _, box in detect(clf, gray, threshold=0.0):
            if all(iou(box, b) < 0.3 for b in boxes):
                x1, y1, x2, y2 = (int(round(v)) for v in clip_box(box, w, h))
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    feats.append(window_feature(gray[y1:y2, x1:x2]))
    return feats


def save_vis(img_path, gt_boxes, dets):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 255, 0), width=2)
    for score, box in dets:
        draw.rectangle(box, outline=(255, 0, 0), width=2)
    img.save(OUT_DIR / f"{img_path.stem}_det.png")


def score_at(per_image, threshold):
    """Aggregate TP/FP/FN over all test images at a given score cutoff."""
    tp = fp = fn = 0
    for boxes, dets in per_image:
        kept = nms([(s, b) for s, b in dets if s >= threshold])
        matched = set()
        for _, box in kept:
            best_i, best_iou = -1, 0.0
            for i, gt in enumerate(boxes):
                if i in matched:
                    continue
                v = iou(box, gt)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_iou >= MATCH_IOU:
                tp += 1
                matched.add(best_i)
            else:
                fp += 1
        fn += len(boxes) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # No true negatives exist in detection, so accuracy is TP / (TP + FP + FN).
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return tp, fp, fn, precision, recall, f1, accuracy


def evaluate(clf, test):
    OUT_DIR.mkdir(exist_ok=True)
    # Collect raw detections once (expensive), then sweep cutoffs cheaply.
    per_image = []
    for img_path, gray, boxes in test:
        dets = detect(clf, gray, threshold=SWEEP_MIN)
        per_image.append((boxes, dets))
        save_vis(img_path, boxes, [(s, b) for s, b in nms(dets) if s >= VIS_THRESHOLD])

    header = f"{'thresh':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'recall':>7} {'F1':>6} {'acc':>6}"
    lines = [header]
    for t in THRESHOLDS:
        tp, fp, fn, p, r, f1, acc = score_at(per_image, t)
        lines.append(f"{t:>7.2f} {tp:>4} {fp:>4} {fn:>4} {p:>6.2f} {r:>7.2f} {f1:>6.2f} {acc:>6.2f}")
    table = "\n".join(lines)
    print(table)
    (OUT_DIR / "metrics.txt").write_text(table + "\n")
    print(f"metrics table saved to {OUT_DIR / 'metrics.txt'}")


def main():
    samples = load_dataset()
    idx = list(range(len(samples)))
    random.Random(SEED).shuffle(idx)
    n_test = max(1, len(samples) // 5)
    test = [samples[i] for i in idx[:n_test]]
    train = [samples[i] for i in idx[n_test:]]
    print(f"{len(train)} train / {len(test)} test images")

    pos = positives(train)
    neg = negatives(train)
    print(f"{len(pos)} positives, {len(neg)} negatives")
    clf = train_svm(pos, neg)

    hard = hard_negatives(clf, train)
    print(f"{len(hard)} hard negatives mined")
    if hard:
        clf = train_svm(pos, neg + hard)

    evaluate(clf, test)
    print(f"annotated detections saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
