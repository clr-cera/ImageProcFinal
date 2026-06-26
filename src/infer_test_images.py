
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

from classifiers import knn_classifier, logreg_classifier, xgboost_classifier
from features import correlogram_features, haralick_features, hog_features
from localizers import (
    hough_ellipse_localizer,
    iou,
    label_boxes,
    segmentation_localizer,
    sliding_window_localizer,
)
from pipeline import DetectionPipeline, _positive_scores

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images_ready"
LABEL_DIR = ROOT / "labels" / "labels"
TEST_DIR = ROOT / "test_images"
OUT_ROOT = ROOT / "results" / "test_images_classical"

WORK_SIZE = (256, 256)  # match the training domain (256x256 processed images)

FEATURES = [hog_features, correlogram_features, haralick_features]
CLASSIFIERS = [logreg_classifier, knn_classifier, xgboost_classifier]
# localizer -> best-F1 threshold taken from results/*/metrics.json
LOCALIZERS = [
    (segmentation_localizer, 0.35),
    (hough_ellipse_localizer, 0.05),
    (sliding_window_localizer, 0.40),
]


def load_dataset():
    images, annotations = [], []
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        stem = re.sub(r"^[0-9a-f]+-", "", label_path.stem)
        img_path = IMG_DIR / f"{stem}.png"
        if not img_path.exists():
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


def nms(boxes, scores, iou_thresh=0.4):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        order = np.array([j for j in order[1:] if iou(boxes[i], boxes[j]) < iou_thresh])
    return keep


def load_test_image(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    work = resize(img, WORK_SIZE, anti_aliasing=True)  # float [0,1]
    return work


def main():
    images, annotations = load_dataset()
    print(f"loaded {len(images)} labeled images for training")
    pipe = DetectionPipeline(images, annotations)

    test_paths = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    test_imgs = {p.name: load_test_image(p) for p in test_paths}
    print(f"{len(test_paths)} test images: {[p.name for p in test_paths]}")

    for localizer, threshold in LOCALIZERS:
        name = localizer.__name__
        print(f"\n=== {name} (threshold {threshold}) ===")

        # train each classifier on the FULL labeled set
        X, y, groups, _ = pipe.build_dataset(localizer, FEATURES)
        if len(np.unique(y)) < 2:
            print("  not enough classes, skipping")
            continue
        models = []
        for clf_fn in CLASSIFIERS:
            model, _ = clf_fn(X, y, X, y)  # eval split unused, just need fitted model
            models.append(model)
        print(f"  trained {len(models)} classifiers on {len(y)} crops "
              f"({int((y == 1).sum())} positive)")

        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for fname, work in tqdm(test_imgs.items(), desc=f"  predict {name}"):
            boxes = localizer(work)
            kept_boxes, kept_scores = [], []
            for box in boxes:
                patch = pipe._crop(work, box)
                if patch is None:
                    continue
                feat = pipe._describe(patch, FEATURES).reshape(1, -1)
                score = float(np.mean([_positive_scores(m, feat)[0] for m in models]))
                if score >= threshold:
                    kept_boxes.append(box)
                    kept_scores.append(score)

            img = Image.fromarray((work * 255).astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(img)
            n_drawn = 0
            if kept_boxes:
                keep = nms(kept_boxes, np.array(kept_scores))
                for i in keep:
                    b = kept_boxes[i]
                    draw.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=2)
                    draw.text((b[0] + 2, b[1] + 2), f"{kept_scores[i]:.2f}", fill=(255, 255, 0))
                    n_drawn += 1
            img.save(out_dir / f"{Path(fname).stem}_det.png")
        print(f"  saved annotated images to {out_dir}")


if __name__ == "__main__":
    main()
