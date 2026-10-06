import os
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Don't compress Image's

PATCH_SIZE = 224
HALF = PATCH_SIZE // 2

histology_dir = "../data/her2st/images/HE"
selection_dir = "../data/her2st/spot-selections"
out_dir = "../data/her2st/patches"

spotid_patch = []

for img_path in sorted(Path(histology_dir).glob("*.jpg")):
    sample_id = img_path.stem
    print(f"sample id: {sample_id}")

    img = Image.open(img_path).convert("RGB")

    selection_path = os.path.join(selection_dir, f"{sample_id}_selection.tsv.gz")
    spots = pd.read_csv(selection_path, sep="\t")
    spots = spots[spots["selected"] == 1]  # only spots actually under tissue

    sample_out_dir = os.path.join(out_dir, sample_id)
    os.makedirs(sample_out_dir, exist_ok=True)

    for _, spot in spots.iterrows():
        cx, cy = int(round(spot["pixel_x"])), int(round(spot["pixel_y"]))
        spot_id = f"{spot['x']}_{spot['y']}"

        patch = img.crop((cx - HALF, cy - HALF, cx + HALF, cy + HALF))

        out_path = os.path.join(sample_out_dir, f"{sample_id}_{spot_id}.png")
        patch.save(out_path)

        spotid_patch.append({"sample": sample_id, "spot": spot_id, "patch_path": out_path})

    img.close()

spotid_patch_df = pd.DataFrame(spotid_patch)
spotid_patch_df.to_csv(os.path.join(out_dir, "spotid_patch.csv"), index=False)