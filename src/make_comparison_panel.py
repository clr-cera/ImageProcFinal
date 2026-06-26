"""Build a side-by-side panel (YOLO + 3 classical models) per test image."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "test_images_comparison"
OUT.mkdir(parents=True, exist_ok=True)

TILE = 256          # each model panel is TILE x TILE
PAD = 8
LABEL_H = 22

MODELS = [
    ("YOLOv8",        RES / "test_images_yolo",                           "{stem}{yext}"),
    ("segmentation",  RES / "test_images_classical/segmentation_localizer", "{stem}_det.png"),
    ("hough_ellipse", RES / "test_images_classical/hough_ellipse_localizer", "{stem}_det.png"),
    ("sliding_window",RES / "test_images_classical/sliding_window_localizer","{stem}_det.png"),
]

TEST_DIR = ROOT / "test_images"


def load_tile(path):
    if path and path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (TILE, TILE), (40, 40, 40))
    img.thumbnail((TILE, TILE))
    canvas = Image.new("RGB", (TILE, TILE), (0, 0, 0))
    canvas.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
    return canvas


def yolo_path(base, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    test_paths = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    n = len(MODELS)
    W = n * TILE + (n + 1) * PAD
    H = TILE + LABEL_H + 2 * PAD

    for tp in test_paths:
        stem = tp.stem
        panel = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (name, base, _) in enumerate(MODELS):
            src = yolo_path(base, stem) if name == "YOLOv8" else base / f"{stem}_det.png"
            tile = load_tile(src)
            x = PAD + i * (TILE + PAD)
            panel.paste(tile, (x, LABEL_H + PAD))
            tw = draw.textlength(name, font=font)
            draw.text((x + (TILE - tw) / 2, PAD // 2), name, fill=(0, 0, 0), font=font)
        out = OUT / f"{stem}_compare.png"
        panel.save(out)
        print(f"saved {out.name}")
    print(f"\n{len(test_paths)} panels in {OUT}")


if __name__ == "__main__":
    main()
