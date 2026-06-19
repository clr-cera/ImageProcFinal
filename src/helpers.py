import numpy as np
from skimage import color

def to_gray(image):
    img = np.asarray(image)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return img


def to_rgb(image):
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3].astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return img
