"""Run the fine-tuned YOLOv8 on test_images/ and save annotated results."""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt"
TEST_DIR = ROOT / "test_images"
OUT_DIR = ROOT / "results" / "test_images_yolo"


def main():
    model = YOLO(str(WEIGHTS))
    results = model.predict(
        str(TEST_DIR),
        conf=0.4,
        save=True,
        project=str(OUT_DIR.parent),
        name=OUT_DIR.name,
        exist_ok=True,
    )
    for r in results:
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"{Path(r.path).name}: {n} cups")
    print(f"\nsaved annotated images to {OUT_DIR}")


if __name__ == "__main__":
    main()
