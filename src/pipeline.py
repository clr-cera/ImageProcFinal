import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import resize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from localizers import label_boxes


def _positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    margin = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-margin))


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


class DetectionPipeline:
    def __init__(self, images, annotations, window=(64, 64)):
        if len(images) != len(annotations):
            raise ValueError("images and annotations must have the same length")
        self.images = list(images)
        self.annotations = list(annotations)
        self.window = window

    def _crop(self, image, box):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return resize(image[y1:y2, x1:x2], self.window, anti_aliasing=True)

    def _describe(self, patch, feature_fns):
        return np.concatenate([np.ravel(fn(patch)) for fn in feature_fns])

    def build_dataset(self, localizer, feature_fns, pos_iou=0.5, neg_iou=0.3):
        X, y, groups, kept_boxes = [], [], [], []
        for img_idx, (image, ann) in enumerate(tqdm(
            list(zip(self.images, self.annotations)), desc="localize+describe"
        )):
            boxes = localizer(image)
            labels = label_boxes(boxes, ann, pos_iou=pos_iou, neg_iou=neg_iou)
            for box, label in zip(boxes, labels):
                if label < 0:
                    continue
                patch = self._crop(image, box)
                if patch is None:
                    continue
                X.append(self._describe(patch, feature_fns))
                y.append(label)
                groups.append(img_idx)
                kept_boxes.append(box)
        return np.array(X), np.array(y), np.array(groups), np.array(kept_boxes, dtype=float)

    def run(
        self,
        localizer,
        feature_fns,
        classifier_fns,
        n_splits=5,
        pos_iou=0.5,
        neg_iou=0.3,
        seed=42,
    ):
        X, y, groups, boxes = self.build_dataset(localizer, feature_fns, pos_iou, neg_iou)
        if X.size == 0 or len(np.unique(y)) < 2:
            raise ValueError("need both positive and negative boxes to train")

        names = [fn.__name__ for fn in classifier_fns]
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_true, fold_vote, fold_groups, fold_boxes = [], [], [], []
        per_clf = {name: [] for name in names}
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            scores = []
            for name, clf_fn in zip(names, classifier_fns):
                model, _ = clf_fn(X_tr, y_tr, X_te, y_te)
                score = _positive_scores(model, X_te)
                scores.append(score)
                per_clf[name].append((score >= 0.5).astype(int))
            vote = (np.mean(scores, axis=0) >= 0.5).astype(int)
            fold_true.append(y_te)
            fold_vote.append(vote)
            fold_groups.append(groups[test_idx])
            fold_boxes.append(boxes[test_idx])

        y_true = np.concatenate(fold_true)
        y_vote = np.concatenate(fold_vote)
        classifier_metrics = {
            name: _metrics(y_true, np.concatenate(folds))
            for name, folds in per_clf.items()
        }
        return {
            "metrics": _metrics(y_true, y_vote),
            "classifier_metrics": classifier_metrics,
            "y_true": y_true,
            "y_pred": y_vote,
            "groups": np.concatenate(fold_groups),
            "boxes": np.concatenate(fold_boxes),
            "n_samples": int(len(y)),
            "n_positive": int((y == 1).sum()),
        }

    def run_and_save(
        self,
        localizer,
        feature_fns,
        classifier_fns,
        n_splits=5,
        pos_iou=0.5,
        neg_iou=0.3,
        seed=42,
    ):
        result = self.run(
            localizer, feature_fns, classifier_fns, n_splits, pos_iou, neg_iou, seed
        )
        feats = "-".join(fn.__name__ for fn in feature_fns)
        clfs = "-".join(fn.__name__ for fn in classifier_fns)
        out_dir = Path(f"{localizer.__name__}_{feats}_{clfs}_detections")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_metrics(out_dir, result)
        self._save_detections(out_dir, result)
        return result

    def _save_metrics(self, out_dir, result):
        lines = [f"samples: {result['n_samples']}  positives: {result['n_positive']}",
                 "ensemble (soft vote):"]
        lines += [f"  {k}: {v:.4f}" for k, v in result["metrics"].items()]
        lines.append("per classifier:")
        for name, m in result["classifier_metrics"].items():
            lines.append(f"  {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
        (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n")
        (out_dir / "metrics.json").write_text(json.dumps({
            "metrics": result["metrics"],
            "classifier_metrics": result["classifier_metrics"],
            "n_samples": result["n_samples"],
            "n_positive": result["n_positive"],
        }, indent=2))

    def _to_pil(self, image):
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255 if arr.max() <= 1 else arr).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def _save_detections(self, out_dir, result):
        groups, y_true, y_pred = result["groups"], result["y_true"], result["y_pred"]
        boxes = result["boxes"]
        for idx, image in enumerate(self.images):
            img = self._to_pil(image)
            draw = ImageDraw.Draw(img)
            for gt in self.annotations[idx]:
                draw.rectangle([gt[0], gt[1], gt[2], gt[3]], outline=(0, 255, 0), width=2)
            in_image = (groups == idx) & (y_pred == 1)
            for box in boxes[in_image & (y_true == 1)]:  # true positives
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 128, 255), width=2)
            for box in boxes[in_image & (y_true == 0)]:  # false positives
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=2)
            img.save(out_dir / f"image_{idx}_det.png")
