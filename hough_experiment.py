"""Hough-ellipse rim proposals + HOG-SVM verification (no deep learning).

Hough-ellipse detection alone finds cup rims with decent recall but terrible
precision -- it fires on every curved edge. Here we use the ellipses purely as
region *proposals* and let the HOG + linear-SVM classifier from hog_svm.py
accept or reject each candidate cup box, combining Hough recall with the
classifier's discrimination.

Pipeline:
  1. Train the HOG+SVM (positives, negatives, hard-negative mining) on the same
     train split hog_svm.py uses.
  2. For each test image: downscale, Canny, and Hough-ellipse to propose rims;
     grow each rim downward by CUP_ASPECT into a candidate cup box.
  3. Score every proposal with the SVM decision function, sweep score cutoffs,
     non-maximum suppress, and report precision/recall/F1/accuracy.
  4. Save annotated images to hough_detections/ (green = ground truth,
     blue = proposed rim, red = accepted cup box).
"""

import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, io
from skimage.feature import canny
from skimage.transform import hough_ellipse, resize

from hog_svm import (
    hard_negatives,
    negatives,
    positives,
    train_svm,
    window_feature,
)

IMG_DIR = Path("images_ready")
LABEL_DIR = Path("labels/labels")
OUT_DIR = Path("hough_detections")

WORK_MAX = 80  # longest side (px) the image is shrunk to before Hough
CANNY_SIGMA = 3.5  # Gaussian blur for Canny; higher = fewer, cleaner edges
MIN_AXIS = 4  # smallest ellipse minor axis to consider (in work-scale px)
MAX_AXIS = 40  # largest ellipse major axis to consider (in work-scale px)
HOUGH_THRESHOLD = 6  # min accumulator votes for an ellipse to be kept
HOUGH_ACCURACY = 12  # bin size for the minor-axis accumulator (speed/precision)
MIN_FLATNESS = 0.15  # reject near-degenerate (line-like) ellipses: b/a >= this
CUP_ASPECT = 1.3  # cup-body height as a multiple of rim width (downward growth)
NMS_IOU = 0.3  # overlap above which cup boxes are merged
MATCH_IOU = 0.5  # overlap required to count a detection as correct
THRESHOLDS = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0]  # SVM operating points to report
VIS_THRESHOLD = 0.0  # SVM cutoff used for the saved annotated images
SEED = 0


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


def ellipse_to_boxes(yc, xc, a, b, orientation, inv_scale, w, h):
    """Map a work-scale ellipse to (rim_box, cup_box) in full-res pixels.

    a/b are the half major/minor axes; orientation is the major-axis angle.
    The axis-aligned half-extents of a rotated ellipse are:
        hx = sqrt((a*cos)^2 + (b*sin)^2),  hy = sqrt((a*sin)^2 + (b*cos)^2)
    """
    cos, sin = math.cos(orientation), math.sin(orientation)
    hx = math.sqrt((a * cos) ** 2 + (b * sin) ** 2) * inv_scale
    hy = math.sqrt((a * sin) ** 2 + (b * cos) ** 2) * inv_scale
    cx, cy = xc * inv_scale, yc * inv_scale

    rim = clip_box([cx - hx, cy - hy, cx + hx, cy + hy], w, h)
    # Grow the rim downward into the cup body; width drives the body height.
    body_h = (2 * hx) * CUP_ASPECT
    cup = clip_box([cx - hx, cy - hy, cx + hx, cy - hy + body_h], w, h)
    return rim, cup


def ellipse_proposals(gray):
    """Propose candidate cup boxes via Hough ellipses. Returns [(cup, rim), ...]."""
    h, w = gray.shape
    scale = WORK_MAX / max(h, w)
    if scale < 1.0:
        work = resize(gray, (max(1, int(h * scale)), max(1, int(w * scale))),
                      anti_aliasing=True)
    else:
        work, scale = gray, 1.0
    inv_scale = 1.0 / scale

    edges = canny(work, sigma=CANNY_SIGMA)
    result = hough_ellipse(
        edges, threshold=HOUGH_THRESHOLD, accuracy=HOUGH_ACCURACY,
        min_size=MIN_AXIS, max_size=MAX_AXIS,
    )
    proposals = []
    for row in result:
        _, yc, xc, a, b, orientation = row  # accumulator votes unused after filter
        major, minor = max(a, b), min(a, b)
        if major <= 0 or minor / major < MIN_FLATNESS:
            continue  # skip line-like / degenerate ellipses
        rim, cup = ellipse_to_boxes(yc, xc, a, b, orientation, inv_scale, w, h)
        if cup[2] - cup[0] < 4 or cup[3] - cup[1] < 4:
            continue
        proposals.append((cup, rim))
    return proposals


def classify(clf, gray, proposals):
    """Score each cup proposal with the HOG SVM. Returns [(svm_score, cup, rim)]."""
    h, w = gray.shape
    feats, kept = [], []
    for cup, rim in proposals:
        x1, y1, x2, y2 = (int(round(v)) for v in cup)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        feats.append(window_feature(gray[y1:y2, x1:x2]))
        kept.append((cup, rim))
    if not feats:
        return []
    scores = clf.decision_function(np.array(feats))
    return [(float(s), cup, rim) for s, (cup, rim) in zip(scores, kept)]


def nms(dets, iou_thr=NMS_IOU):
    keep = []
    for det in sorted(dets, key=lambda d: d[0], reverse=True):
        if all(iou(det[1], kept[1]) < iou_thr for kept in keep):
            keep.append(det)
    return keep


def save_vis(img_path, gt_boxes, dets):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in gt_boxes:
        draw.rectangle(box, outline=(0, 255, 0), width=2)
    for _, cup, rim in dets:
        draw.rectangle(rim, outline=(0, 128, 255), width=1)
        draw.rectangle(cup, outline=(255, 0, 0), width=2)
    img.save(OUT_DIR / f"{img_path.stem}_hough.png")


def score_at(per_image, threshold):
    """Aggregate TP/FP/FN over all test images at a given SVM score cutoff."""
    tp = fp = fn = 0
    for boxes, dets in per_image:
        kept = nms([d for d in dets if d[0] >= threshold])
        matched = set()
        for _, cup, _ in kept:
            best_i, best_iou = -1, 0.0
            for i, gt in enumerate(boxes):
                if i in matched:
                    continue
                v = iou(cup, gt)
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
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return tp, fp, fn, precision, recall, f1, accuracy


def evaluate(clf, test):
    OUT_DIR.mkdir(exist_ok=True)
    # Score all proposals once, then sweep SVM cutoffs cheaply.
    per_image = []
    for img_path, gray, boxes in test:
        dets = classify(clf, gray, ellipse_proposals(gray))
        per_image.append((boxes, dets))
        vis = nms([d for d in dets if d[0] >= VIS_THRESHOLD])
        save_vis(img_path, boxes, vis)
        print(f"  {img_path.name}: {len(dets)} proposals, {len(vis)} accepted")

    header = f"{'thresh':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>6} {'recall':>7} {'F1':>6} {'acc':>6}"
    lines = [header]
    for t in THRESHOLDS:
        tp, fp, fn, p, r, f1, acc = score_at(per_image, t)
        lines.append(f"{t:>7.2f} {tp:>4} {fp:>4} {fn:>4} {p:>6.2f} {r:>7.2f} {f1:>6.2f} {acc:>6.2f}")
    table = "\n".join(lines)
    print(table)
    (OUT_DIR / "metrics.txt").write_text(table + "\n")
    print(f"metrics table saved to {OUT_DIR / 'metrics.txt'}")
    print(f"annotated detections saved to {OUT_DIR}/")


def main():
    samples = load_dataset()
    idx = list(range(len(samples)))
    random.Random(SEED).shuffle(idx)
    n_test = max(1, len(samples) // 5)
    test = [samples[i] for i in idx[:n_test]]
    train = [samples[i] for i in idx[n_test:]]
    print(f"{len(train)} train / {len(test)} test images (same split as hog_svm.py)")

    pos = positives(train)
    neg = negatives(train)
    print(f"{len(pos)} positives, {len(neg)} negatives")
    clf = train_svm(pos, neg)
    hard = hard_negatives(clf, train)
    print(f"{len(hard)} hard negatives mined")
    if hard:
        clf = train_svm(pos, neg + hard)

    evaluate(clf, test)


if __name__ == "__main__":
    main()
