import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog

from helpers import to_gray, to_rgb


def hog_features(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    transform_sqrt=True,
):
    gray = to_gray(image)
    return hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        transform_sqrt=transform_sqrt,
        feature_vector=True,
        channel_axis=None,
    )


def _quantize(rgb, bins):
    q = np.clip((rgb * bins).astype(np.int64), 0, bins - 1)
    return q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]


def _ring_offsets(d):
    offs = []
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            if max(abs(dy), abs(dx)) == d:
                offs.append((dy, dx))
    return offs


def correlogram_features(image, bins=4, distances=(1, 3, 5, 7)):
    rgb = to_rgb(image)
    labels = _quantize(rgb, bins)
    h, w = labels.shape
    n_colors = bins ** 3
    feat = []
    for d in distances:
        same = np.zeros(n_colors, dtype=np.float64)
        total = np.zeros(n_colors, dtype=np.float64)
        for dy, dx in _ring_offsets(d):
            base = labels[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            neigh = labels[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
            base_f = base.ravel()
            total += np.bincount(base_f, minlength=n_colors)
            same += np.bincount(base_f[base_f == neigh.ravel()], minlength=n_colors)
        feat.append(np.divide(same, total, out=np.zeros_like(same), where=total > 0))
    return np.concatenate(feat)


HARALICK_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy",
                  "correlation", "ASM")


def haralick_features(
    image,
    levels=16,
    distances=(1, 2, 3),
    angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    props=HARALICK_PROPS,
):
    gray = to_gray(image)
    q = np.clip((gray * levels).astype(np.uint8), 0, levels - 1)
    glcm = graycomatrix(
        q, distances=list(distances), angles=list(angles),
        levels=levels, symmetric=True, normed=True,
    )
    feat = []
    for prop in props:
        # graycoprops -> (n_distances, n_angles); average over angles.
        feat.append(graycoprops(glcm, prop).mean(axis=1))
    return np.concatenate(feat)
