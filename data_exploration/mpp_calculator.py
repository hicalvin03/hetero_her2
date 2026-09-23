import pandas as pd
import numpy as np
import glob
import os

SPOT_SPACING_UM = 200.0  # known physical spacing between centre to centre of spots

def compute_mpp(selection_path, n_pairs=1000, seed=0):
    df = pd.read_csv(selection_path, sep="\t")
    
    ax, ay = df["new_x"].values, df["new_y"].values   # array-grid coords
    px, py = df["pixel_x"].values, df["pixel_y"].values  # pixel coords
    
    n = len(df)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    
    # Calculate distance between pairs for pixel and array
    dist_array = np.hypot(ax[i] - ax[j], ay[i] - ay[j])
    dist_pixel = np.hypot(px[i] - px[j], py[i] - py[j])
    
    valid = dist_array > 0
    ratios = dist_pixel[valid] / dist_array[valid]  # pixel_distance/array distance how many pixels per grid step
    
    pixels_per_array_unit = np.median(ratios)
    mpp = SPOT_SPACING_UM / pixels_per_array_unit
    return mpp

results = []
for path in glob.glob("../data/her2st/spot-selections/*.tsv.gz"):

    sample_id = os.path.basename(path).split("_selection")[0]
    mpp = compute_mpp(path)
    results.append({"sample": sample_id, "mpp": mpp})
    

mpp_table = pd.DataFrame(results)
mpp_table.to_csv("her2st_mpp_per_sample.csv", index=False)