"""Felzenszwalb region proposals + (color correlogram + HOG) SVM cup detector.

Where hough_experiment.py proposes cup boxes from Hough-ellipse rims, here the
object proposals come from Felzenszwalb-Huttenlocher graph-based segmentation:
each segment's bounding box is a candidate object. Segmentation alone groups
pixels but says nothing about *what* they are, so a linear SVM accepts or
rejects each box. The descriptor mixes shape and color: a grayscale HOG vector
(cup silhouette / rim edges) concatenated with an RGB color autocorrelogram
(the spatial color texture of porcelain, handles, drinks).

Pipeline:
  1. Train the SVM on the same train split the other experiments use:
     positive cup crops (+ flips), random negatives, then hard-negative mining
     from the segmentation proposals themselves.
  2. For each test image: run Felzenszwalb at a few scales, turn every segment
     into a candidate box, and describe each box with HOG + correlogram.
  3. Score every proposal with the SVM, sweep score cutoffs, non-maximum
     suppress, and report precision/recall/F1/accuracy in 5-fold CV.
  4. Save annotated images to felze_detections/ (green = ground truth,
     blue = raw proposal, red = accepted cup box).
"""

import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from skimage import color, io
from skimage.feature import hog
from skimage.segmentation import felzenszwalb
from skimage.transform import resize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

IMG_DIR = Path("images_ready")
LABEL_DIR = Path("labels/labels")
OUT_DIR = Path("felze_detections")

WINDOW = (64, 64)  # (height, width) every proposal is resized to before describing
HOG_KW = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    transform_sqrt=True,
    feature_vector=True,
    channel_axis=None,
)

CORR_BINS = 4  # quantization levels per RGB channel -> CORR_BINS**3 colors
CORR_DISTANCES = (1, 3, 5, 7)  # Chebyshev pixel distances for the autocorrelogram

# Felzenszwalb is run at several (scale, sigma, min_size) settings so a cup that
# merges into the background at one scale is isolated at another.
FELZ_PARAMS = [
    dict(scale=100, sigma=0.5, min_size=50),
    dict(scale=300, sigma=0.7, min_size=100),
    dict(scale=600, sigma=0.8, min_size=200),
]
MIN_PROP = 16  # smallest proposal side (px) worth describing
MAX_PROP_FRAC = 0.95  # drop near-whole-image segments (background blobs)

NEG_PER_IMAGE = 20  # random negative windows sampled per training image
MAX_FELZ_NEG = 2000  # cap on mined segmentation hard negatives (rest are redundant)
NMS_IOU = 0.3  # overlap above which cup boxes are merged
MATCH_IOU = 0.5  # overlap required to count a detection as correct
THRESHOLDS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0]  # SVM operating points to report
VIS_THRESHOLD = 0.0  # SVM cutoff used for the saved annotated images
N_FOLDS = 5  # cross-validation folds (every image is tested exactly once)
SEED = 42

RNG = np.random.default_rng(SEED)


def load_rgb(path):
    """Load an image as a float RGB array in [0, 1], dropping any alpha."""
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    return img.astype(np.float64) / 255.0


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
    """Return [(image_path, rgb_array, [boxes_xyxy_pixels]), ...]."""
    samples = []
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        stem = re.sub(r"^[0-9a-f]+-", "", label_path.stem)  # <hash>-image_N -> image_N
        img_path = IMG_DIR / f"{stem}.png"
        if not img_path.exists():
            print(f"skip {label_path.name}: no matching image")
            continue
        rgb = load_rgb(img_path)
        h, w = rgb.shape[:2]
        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            boxes.append(clip_box(box, w, h))
        samples.append((img_path, rgb, boxes))
    return samples


def quantize(rgb):
    """Map an RGB image in [0, 1] to a per-pixel color index in [0, CORR_BINS**3)."""
    q = np.clip((rgb * CORR_BINS).astype(np.int64), 0, CORR_BINS - 1)
    return q[..., 0] * CORR_BINS * CORR_BINS + q[..., 1] * CORR_BINS + q[..., 2]


def _ring_offsets(d):
    """(dy, dx) offsets whose Chebyshev distance to the origin equals exactly d."""
    offs = []
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            if max(abs(dy), abs(dx)) == d:
                offs.append((dy, dx))
    return offs


def correlogram(rgb):
    """Color autocorrelogram: per color, the fraction of pixels at Chebyshev
    distance d that share that color, for each d in CORR_DISTANCES.

    Returns a flat vector of length CORR_BINS**3 * len(CORR_DISTANCES).
    """
    labels = quantize(rgb)
    h, w = labels.shape
    n_colors = CORR_BINS ** 3
    feat = []
    for d in CORR_DISTANCES:
        same = np.zeros(n_colors, dtype=np.float64)
        total = np.zeros(n_colors, dtype=np.float64)
        for dy, dx in _ring_offsets(d):
            # Overlap region where both the base pixel and its (dy, dx) neighbour exist.
            base = labels[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            neigh = labels[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
            base_f = base.ravel()
            total += np.bincount(base_f, minlength=n_colors)
            same += np.bincount(base_f[base_f == neigh.ravel()], minlength=n_colors)
        feat.append(np.divide(same, total, out=np.zeros_like(same), where=total > 0))
    return np.concatenate(feat)


def window_feature(rgb_patch):
    """HOG (on grayscale) + color autocorrelogram of a crop resized to WINDOW."""
    patch = resize(rgb_patch, WINDOW, anti_aliasing=True)
    hog_feat = hog(color.rgb2gray(patch), **HOG_KW)
    return np.concatenate([hog_feat, correlogram(patch)])


def felz_proposals(rgb):
    """Bounding boxes of Felzenszwalb segments across FELZ_PARAMS. Returns [box_xyxy]."""
    h, w = rgb.shape[:2]
    max_area = MAX_PROP_FRAC * w * h
    seen, proposals = set(), []
    for params in FELZ_PARAMS:
        seg = felzenszwalb(rgb, **params)
        for label in range(seg.max() + 1):
            ys, xs = np.where(seg == label)
            if xs.size == 0:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < MIN_PROP or bh < MIN_PROP or bw * bh > max_area:
                continue
            key = tuple(int(round(v)) for v in box)  # dedup identical boxes across scales
            if key not in seen:
                seen.add(key)
                proposals.append(box)
    return proposals


def crop(rgb, box):
    """Integer-clipped RGB crop, or None if it is too small to describe."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return rgb[y1:y2, x1:x2]


def classify(clf, rgb, proposals):
    """Score each proposal with the SVM. Returns [(svm_score, box), ...]."""
    feats, kept = [], []
    for box in proposals:
        patch = crop(rgb, box)
        if patch is None:
            continue
        feats.append(window_feature(patch))
        kept.append(box)
    if not feats:
        return []
    scores = clf.decision_function(np.array(feats))
    return [(float(s), box) for s, box in zip(scores, kept)]


def positives(samples):
    feats = []
    for _, rgb, boxes in tqdm(samples, desc="positives"):
        for box in boxes:
            patch = crop(rgb, box)
            if patch is None:
                continue
            feats.append(window_feature(patch))
            feats.append(window_feature(patch[:, ::-1]))  # horizontal flip
    return feats


def negatives(samples, n_per_image=NEG_PER_IMAGE):
    feats = []
    for _, rgb, boxes in tqdm(samples, desc="negatives"):
        h, w = rgb.shape[:2]
        count = attempts = 0
        while count < n_per_image and attempts < n_per_image * 20:
            attempts += 1
            size = int(RNG.integers(32, min(h, w)))
            x1 = int(RNG.integers(0, w - size + 1))
            y1 = int(RNG.integers(0, h - size + 1))
            cand = [x1, y1, x1 + size, y1 + size]
            if all(iou(cand, b) < 0.1 for b in boxes):
                feats.append(window_feature(rgb[y1:y1 + size, x1:x1 + size]))
                count += 1
    return feats


def felz_hard_negatives(clf, samples, threshold=0.0):
    """Segment proposals the SVM accepts but that overlap no real cup.

    These are exactly the background blobs (plates, tables, shadows) the
    classifier currently mistakes for cups, so they make informative negatives.
    """
    feats = []
    for _, rgb, boxes in tqdm(samples, desc="felz hard negatives"):
        h, w = rgb.shape[:2]
        for s, box in classify(clf, rgb, felz_proposals(rgb)):
            if s < threshold:
                continue
            if all(iou(box, b) < 0.3 for b in boxes):
                patch = crop(rgb, clip_box(box, w, h))
                if patch is not None:
                    feats.append(window_feature(patch))
    # Proposals are highly redundant; a random cap keeps the SVM balanced and
    # lets liblinear converge quickly instead of grinding to max_iter.
    if len(feats) > MAX_FELZ_NEG:
        feats = random.Random(SEED).sample(feats, MAX_FELZ_NEG)
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


def train_classifier(train):
    """Run the full training recipe (SVM + one round of hard-negative mining)."""
    pos = positives(train)
    neg = negatives(train)
    clf = train_svm(pos, neg)
    hard = felz_hard_negatives(clf, train)
    tqdm.write(f"  {len(pos)} pos, {len(neg)} neg, {len(hard)} felz hard negatives")
    if hard:
        clf = train_svm(pos, neg + hard)
    return clf


def nms(dets, iou_thr=NMS_IOU):
    keep = []
    for score, box in sorted(dets, key=lambda d: d[0], reverse=True):
        if all(iou(box, kb) < iou_thr for _, kb in keep):
            keep.append((score, box))
    return keep


def save_vis(img_path, gt_boxes, proposals, dets):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in proposals:
        draw.rectangle(box, outline=(0, 128, 255), width=1)
    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 255, 0), width=2)
    for score, box in dets:
        draw.rectangle(box, outline=(255, 0, 0), width=2)
    img.save(OUT_DIR / f"{img_path.stem}_felz.png")


def score_at(per_image, threshold):
    """Aggregate TP/FP/FN over all test images at a given SVM score cutoff."""
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


def report(per_image):
    """Sweep SVM cutoffs over the pooled cross-validation detections."""
    header = f"{'thresh':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'recall':>7} {'F1':>6} {'acc':>6}"
    lines = [header]
    for t in THRESHOLDS:
        tp, fp, fn, p, r, f1, acc = score_at(per_image, t)
        lines.append(f"{t:>7.2f} {tp:>4} {fp:>4} {fn:>4} {p:>6.2f} {r:>7.2f} {f1:>6.2f} {acc:>6.2f}")
    table = "\n".join(lines)
    print(table)
    (OUT_DIR / "metrics.txt").write_text(table + "\n")


def main():
    samples = load_dataset()
    OUT_DIR.mkdir(exist_ok=True)
    idx = list(range(len(samples)))
    random.Random(SEED).shuffle(idx)
    folds = [idx[k::N_FOLDS] for k in range(N_FOLDS)]  # each image in one fold

    # Cross-validation: train on the other folds, test on this fold, and pool
    # every image's detections so the final metrics use the whole dataset once.
    per_image = []
    for k, fold in enumerate(folds):
        test_ids = set(fold)
        test = [samples[i] for i in fold]
        train = [samples[i] for i in idx if i not in test_ids]
        print(f"fold {k + 1}/{N_FOLDS}: {len(train)} train / {len(test)} test images")
        clf = train_classifier(train)
        for img_path, rgb, boxes in tqdm(test, desc=f"fold {k + 1} detecting"):
            proposals = felz_proposals(rgb)
            dets = classify(clf, rgb, proposals)
            per_image.append((boxes, dets))
            kept = [(s, b) for s, b in nms(dets) if s >= VIS_THRESHOLD]
            save_vis(img_path, boxes, proposals, kept)

    report(per_image)
    print(f"{N_FOLDS}-fold CV over {len(per_image)} images; "
          f"metrics saved to {OUT_DIR / 'metrics.txt'}, annotated images in {OUT_DIR}/")


if __name__ == "__main__":
    main()
