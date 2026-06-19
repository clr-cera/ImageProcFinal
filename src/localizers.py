import math

import numpy as np
from skimage.feature import canny
from skimage.segmentation import felzenszwalb
from skimage.transform import hough_ellipse, resize
from helpers import to_gray, to_rgb

FELZ_PARAMETERS = (
    dict(scale=100, sigma=0.5, min_size=50),
    dict(scale=300, sigma=0.7, min_size=100),
    dict(scale=600, sigma=0.8, min_size=200),
)

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


def label_boxes(boxes, annotations, pos_iou=0.5, neg_iou=0.3, ignore_label=-1):
    labels = []
    for box in boxes:
        best = max((iou(box, gt) for gt in annotations), default=0.0)
        if best >= pos_iou:
            labels.append(1)
        elif best <= neg_iou:
            labels.append(0)
        else:
            labels.append(ignore_label)
    return labels


def sliding_window_localizer(
    image,
    window=(64, 64),
    pyramid_downscale=1.2,
    step=12,
):
    gray = to_gray(image)
    h0, w0 = gray.shape
    win_h, win_w = window
    boxes = []
    scale = 1.0
    while True:
        img = gray if scale == 1.0 else resize(
            gray, (max(1, int(h0 * scale)), max(1, int(w0 * scale))), anti_aliasing=True
        )
        sh, sw = img.shape
        if sh < win_h or sw < win_w:
            break
        inv = 1.0 / scale
        for y in range(0, sh - win_h + 1, step):
            for x in range(0, sw - win_w + 1, step):
                boxes.append([x * inv, y * inv, (x + win_w) * inv, (y + win_h) * inv])
        scale /= pyramid_downscale
    return boxes


def hough_ellipse_localizer(
    image,
    work_max=80,
    canny_sigma=3.5,
    min_axis=4,
    max_axis=40,
    hough_threshold=6,
    hough_accuracy=12,
    min_flatness=0.15,
    cup_aspect=1.3,
    min_box=4,
):
    gray = to_gray(image)
    h, w = gray.shape
    scale = work_max / max(h, w)
    if scale < 1.0:
        work = resize(gray, (max(1, int(h * scale)), max(1, int(w * scale))),
                      anti_aliasing=True)
    else:
        work, scale = gray, 1.0
    inv_scale = 1.0 / scale

    edges = canny(work, sigma=canny_sigma)
    result = hough_ellipse(
        edges, threshold=hough_threshold, accuracy=hough_accuracy,
        min_size=min_axis, max_size=max_axis,
    )
    boxes = []
    for row in result:
        _, yc, xc, a, b, orientation = row
        major, minor = max(a, b), min(a, b)
        if major <= 0 or minor / major < min_flatness:
            continue
        cos, sin = math.cos(orientation), math.sin(orientation)
        hx = math.sqrt((a * cos) ** 2 + (b * sin) ** 2) * inv_scale
        hy = math.sqrt((a * sin) ** 2 + (b * cos) ** 2) * inv_scale
        cx, cy = xc * inv_scale, yc * inv_scale
        body_h = (2 * hx) * cup_aspect
        cup = clip_box([cx - hx, cy - hy, cx + hx, cy - hy + body_h], w, h)
        if cup[2] - cup[0] < min_box or cup[3] - cup[1] < min_box:
            continue
        boxes.append(cup)
    return boxes


def segmentation_localizer(
    image,
    felz_params=FELZ_PARAMETERS,
    min_prop=16,
    max_prop_frac=0.95,
):
    rgb = to_rgb(image)
    h, w = rgb.shape[:2]
    max_area = max_prop_frac * w * h
    seen, boxes = set(), []
    for params in felz_params:
        seg = felzenszwalb(rgb, **params)
        for label in range(seg.max() + 1):
            ys, xs = np.where(seg == label)
            if xs.size == 0:
                continue
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < min_prop or bh < min_prop or bw * bh > max_area:
                continue
            key = tuple(int(round(v)) for v in box)
            if key not in seen:
                seen.add(key)
                boxes.append(box)
    return boxes
