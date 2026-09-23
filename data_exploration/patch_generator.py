import pandas as pd
import os
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader
from pathlib import Path
import torch
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0)) 


HOVERNET_MPP = 0.25 # 40x magnification
PATCH_SIZE = 256

Patient_A_mpp = 0.92
Nonpatient_A_mpp = 0.69


mpp_table = pd.read_csv("her2st_mpp_per_sample.csv").set_index("sample")

histology_dir = "../data/her2st/images/HE"
selection_dir = "../data/her2st/spot-selections"
out_dir = "../data/her2st/patches"

spotid_patch = []

image_folder = Path(histology_dir)

for img_path in image_folder.glob('*.jpg'):

    sample_id = img_path.stem
    print(f"sample id: {sample_id}")

    # noticed that mpp for patient A is different from rest.
    mpp = Nonpatient_A_mpp
    if sample_id[0] == "A":
        mpp = Patient_A_mpp
    
    wsi = WSIReader.open(input_img=img_path, mpp=(mpp, mpp))
    selection_path = os.path.join(selection_dir, f"{sample_id}_selection.tsv.gz")
    spots = pd.read_csv(selection_path, sep="\t")
    spots = spots[spots["selected"] == 1]  # only spots actually under tissue

    sample_out_dir = os.path.join(out_dir, sample_id)
    os.makedirs(sample_out_dir, exist_ok=True)

    for _, spot in spots.iterrows():
        x, y = spot["pixel_x"], spot["pixel_y"]
        spot_id = f"{spot['x']}_{spot['y']}"

        patch = wsi.read_rect(
            location=(x, y),
            size=(PATCH_SIZE, PATCH_SIZE),
            resolution=HOVERNET_MPP,
            units="mpp",
        )

        out_path = os.path.join(sample_out_dir, f"{sample_id}_{spot_id}.png")
        Image.fromarray(patch).save(out_path)

        spotid_patch.append({"sample": sample_id, "spot": spot_id, "patch_path": out_path})

spotid_patch_df = pd.DataFrame(spotid_patch)
spotid_patch_df.to_csv(os.path.join(out_dir, "spotid_patch.csv"), index=False)


from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

segmentor = MultiTaskSegmentor(
    model="hovernet_fast-pannuke",
    num_workers=4,
    batch_size=4,
    device="cuda",
)

output = segmentor.run(
    images=spotid_patch_df["patch_path"].tolist(),
    save_dir="../data/her2st/hovernet_output",
    patch_mode=True,
    output_type="dict",
)