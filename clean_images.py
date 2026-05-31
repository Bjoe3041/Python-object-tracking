import os
import numpy as np
import torch
from PIL import Image

# ---------------- CONFIG ----------------
OBJ_DIR = "objects"
OUT_DIR = "objects_cleaned"

# How aggressive to be with color removal
SAT_THRESHOLD = 0.15      # pixels below this saturation are checked for B&W removal
WHITE_THRESHOLD = 0.85    # brightness above this = white (removed)
BLACK_THRESHOLD = 0.12    # brightness below this = black (removed)

# Hue ranges for green and blue (0-360)
GREEN_HUE = (60, 165)
BLUE_HUE  = (185, 270)
# ----------------------------------------

def remove_background(img: Image.Image) -> Image.Image:
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)
    delta = maxc - minc

    # --- Value (brightness) ---
    value = maxc

    # --- Saturation ---
    saturation = np.where(maxc > 0, delta / maxc, 0)

    # --- Hue ---
    hue = np.zeros_like(maxc)
    mask_r = (maxc == r) & (delta > 0)
    mask_g = (maxc == g) & (delta > 0)
    mask_b = (maxc == b) & (delta > 0)
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    hue[mask_g] =  60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
    hue[mask_b] =  60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240

    # --- Masks ---
    is_white  = (saturation < SAT_THRESHOLD) & (value > WHITE_THRESHOLD)
    is_black  = (saturation < SAT_THRESHOLD) & (value < BLACK_THRESHOLD)
    is_green  = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]) & (saturation > SAT_THRESHOLD)
    is_blue   = (hue >= BLUE_HUE[0])  & (hue <= BLUE_HUE[1])  & (saturation > SAT_THRESHOLD)

    remove = is_white | is_black | is_green | is_blue

    # --- Apply as alpha ---
    rgba = np.array(img.convert("RGBA"))
    rgba[:,:,3][remove] = 0

    return Image.fromarray(rgba, "RGBA")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(OBJ_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Processing {len(files)} images...")

    for i, fname in enumerate(files):
        src = os.path.join(OBJ_DIR, fname)
        # Always save as PNG to preserve alpha channel
        out_name = os.path.splitext(fname)[0] + ".png"
        dst = os.path.join(OUT_DIR, out_name)

        img = Image.open(src)
        result = remove_background(img)
        result.save(dst)

        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    print(f"Done. Cleaned images saved to '{OUT_DIR}/'")