"""Convert all images in images/ to 256x256 PNGs in images_ready/."""

from pathlib import Path

import pillow_avif  # noqa: F401  (registers AVIF support with Pillow)
from PIL import Image

SRC_DIR = Path("images")
DST_DIR = Path("images_ready")
SIZE = (256, 256)


def main() -> None:
    DST_DIR.mkdir(exist_ok=True)

    images = sorted(p for p in SRC_DIR.iterdir() if p.is_file())
    for path in images:
        try:
            with Image.open(path) as img:
                img = img.convert("RGBA")
                img = img.resize(SIZE, Image.LANCZOS)
                out_path = DST_DIR / f"{path.stem}.png"
                img.save(out_path, "PNG")
                print(f"{path.name} -> {out_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {path.name}: {exc}")


if __name__ == "__main__":
    main()
