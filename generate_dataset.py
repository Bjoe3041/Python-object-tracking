import os
import random
import shutil
from PIL import Image

# ---------------- CONFIG ----------------
NUM_IMAGES = 5000                # how many synthetic examples to generate
OBJ_DIR = "objects"             # folder with object images
BG_DIR = "backgrounds"          # folder with background images
OUT_DIR = "combination_dataset" # output dataset folder
REAL_DIR = "uav_dataset"        # root of the real roboflow dataset
TRAIN_SPLIT = 0.8               # % train / % val split for synthetic
CLASS_NAMES = ["drone"]
# ----------------------------------------

def prepare_folders():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)


def get_random_image(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return os.path.join(folder, random.choice(files))


def place_object_on_background(obj_path, bg_path, out_path_img, out_path_label):
    bg = Image.open(bg_path).convert("RGB")
    obj = Image.open(obj_path).convert("RGBA")

    bw, bh = bg.size

    scale_factor = random.uniform(0.1, 0.3)
    new_w = int(bw * scale_factor)
    aspect_ratio = obj.width / obj.height
    new_h = int(new_w / aspect_ratio)
    obj = obj.resize((new_w, new_h), Image.LANCZOS)

    max_x = bw - new_w
    max_y = bh - new_h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    bg.paste(obj, (x, y), obj)
    bg.save(out_path_img)

    x_center = (x + new_w / 2) / bw
    y_center = (y + new_h / 2) / bh
    w_norm = new_w / bw
    h_norm = new_h / bh

    with open(out_path_label, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def generate_synthetic():
    print("Generating synthetic images...")
    for i in range(NUM_IMAGES):
        split = "train" if i < int(NUM_IMAGES * TRAIN_SPLIT) else "val"

        obj_path = get_random_image(OBJ_DIR)
        bg_path = get_random_image(BG_DIR)

        out_img   = os.path.join(OUT_DIR, "images", split, f"syn_{i:05d}.jpg")
        out_label = os.path.join(OUT_DIR, "labels", split, f"syn_{i:05d}.txt")

        place_object_on_background(obj_path, bg_path, out_img, out_label)

        if i % 100 == 0:
            print(f"  {i}/{NUM_IMAGES}")


def copy_real_dataset():
    """
    Copies images + labels from the real roboflow dataset into the combined folder.
    Expects structure:
        uav_dataset/train/images/*.jpg
        uav_dataset/train/labels/*.txt
        uav_dataset/valid/images/*.jpg   (roboflow uses 'valid' not 'val')
        uav_dataset/valid/labels/*.txt
    """
    print("Copying real dataset...")

    # Roboflow uses 'valid', we use 'val'
    split_map = {"train": "train", "valid": "val"}

    for real_split, out_split in split_map.items():
        img_src = os.path.join(REAL_DIR, real_split, "images")
        lbl_src = os.path.join(REAL_DIR, real_split, "labels")

        if not os.path.exists(img_src):
            print(f"  Skipping {real_split} — folder not found: {img_src}")
            continue

        images = [f for f in os.listdir(img_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fname in images:
            stem = os.path.splitext(fname)[0]

            src_img = os.path.join(img_src, fname)
            src_lbl = os.path.join(lbl_src, stem + ".txt")

            dst_img = os.path.join(OUT_DIR, "images", out_split, f"real_{fname}")
            dst_lbl = os.path.join(OUT_DIR, "labels", out_split, f"real_{stem}.txt")

            shutil.copy2(src_img, dst_img)

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # No label = background-only image, write empty label file
                open(dst_lbl, "w").close()

        print(f"  Copied {len(images)} images from {real_split} → {out_split}")


def write_yaml():
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names: " + str(CLASS_NAMES) + "\n")


if __name__ == "__main__":
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    prepare_folders()
    generate_synthetic()
    copy_real_dataset()
    write_yaml()

    # Print summary
    for split in ["train", "val"]:
        n = len(os.listdir(os.path.join(OUT_DIR, "images", split)))
        print(f"  {split}: {n} images")

    print("Done — combined dataset ready.")