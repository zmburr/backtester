"""Build Method's icon set from the logo artwork.

    venv\\Scripts\\python.exe method/brand/make_icons.py

logo-source.png is the master: a violet M on a near-black tile (1254 px,
supplied by the user on 2026-10-15). The app icons are that tile resized; the
M sits inside the central 80% circle, so it also serves as the maskable icon.
mark.png is the M alone, glow included, on transparency, for the sidebar and
top bar. Outputs go to method/static/icons/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SRC = HERE / "logo-source.png"
OUT = HERE.parent / "static" / "icons"
BG = np.array([8, 7, 15], dtype=float)       # the tile's corner colour
MARK_BOX = (236, 272, 1018, 894)               # the M plus its glow, measured on the master


def color_to_alpha(img: Image.Image) -> Image.Image:
    """Lift the tile colour out so the M and its glow sit on any dark surface."""
    a = np.asarray(img.convert("RGB")).astype(float)
    span = np.where(a >= BG, 255 - BG, BG)
    alpha = np.clip((np.abs(a - BG) / np.maximum(span, 1)).max(axis=2), 0, 1)
    safe = np.where(alpha > 0, alpha, 1)[..., None]
    rgb = np.clip((a - BG * (1 - alpha[..., None])) / safe, 0, 255)
    out = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def rounded(img: Image.Image, radius: float) -> Image.Image:
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=radius * img.width, fill=255)
    img.putalpha(mask)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = Image.open(SRC).convert("RGB")
    sized = lambda n: src.resize((n, n), Image.LANCZOS)
    sized(180).save(OUT / "apple-touch-icon.png")                 # iOS rounds it itself
    sized(192).save(OUT / "icon-192.png")
    sized(512).save(OUT / "icon-512.png")
    sized(512).save(OUT / "icon-maskable-512.png")
    # favicon: crop in on the M so it reads at 32 px, then round the corners
    x0, y0, x1, y1 = MARK_BOX
    pad = 70
    side = max(x1 - x0, y1 - y0) + 2 * pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    fav = src.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    rounded(fav.resize((64, 64), Image.LANCZOS), 0.22).resize((32, 32), Image.LANCZOS).save(OUT / "favicon-32.png")
    rounded(fav.resize((180, 180), Image.LANCZOS), 0.22).save(OUT / "favicon-180.png")
    mark = color_to_alpha(src.crop(MARK_BOX))
    h = 120
    mark.resize((round(mark.width * h / mark.height), h), Image.LANCZOS).save(OUT / "mark.png")
    for stale in ("favicon.svg", "glyph.svg"):
        (OUT / stale).unlink(missing_ok=True)
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
