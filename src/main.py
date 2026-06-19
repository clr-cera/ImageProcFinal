import re
from pathlib import Path

from skimage import io
from tqdm import tqdm

from classifiers import knn_classifier, logreg_classifier, xgboost_classifier
from features import correlogram_features, haralick_features, hog_features
from localizers import (
    hough_ellipse_localizer,
    segmentation_localizer,
    sliding_window_localizer,
)
from pipeline import DetectionPipeline

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images_ready"
LABEL_DIR = ROOT / "labels" / "labels"

LOCALIZERS = [
    segmentation_localizer,
    hough_ellipse_localizer,
    sliding_window_localizer,
]
FEATURES = [hog_features, correlogram_features, haralick_features]
CLASSIFIERS = [logreg_classifier, knn_classifier, xgboost_classifier]


def load_dataset():
    images, annotations = [], []
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        stem = re.sub(r"^[0-9a-f]+-", "", label_path.stem)
        img_path = IMG_DIR / f"{stem}.png"
        if not img_path.exists():
            print(f"skip {label_path.name}: no matching image")
            continue
        image = io.imread(img_path)
        h, w = image.shape[:2]
        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append([
                (cx - bw / 2) * w, (cy - bh / 2) * h,
                (cx + bw / 2) * w, (cy + bh / 2) * h,
            ])
        images.append(image)
        annotations.append(boxes)
    return images, annotations


def main():
    images, annotations = load_dataset()
    print(f"loaded {len(images)} images")
    pipeline = DetectionPipeline(images, annotations)

    for localizer in tqdm(LOCALIZERS, desc="Localizers"):
        print(f"\n=== {localizer.__name__} ===")
        result = pipeline.run_and_save(localizer, FEATURES, CLASSIFIERS, threshold="best-f1")
        print(f"best threshold: {result['threshold']}")
        print({k: round(v, 4) for k, v in result["metrics"].items()})


if __name__ == "__main__":
    main()
