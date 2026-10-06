from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../data/her2st/patches/spotid_patch.csv")

image_paths = df["patch_path"].tolist()

# Init the processor and model
processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
model = AutoModel.from_pretrained("owkin/phikon-v2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

BATCH_SIZE = 32
all_features = []


for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Running Phikon"):
    batch_paths = image_paths[i : i + BATCH_SIZE]
    
    # Load 1 batch at a time
    batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
    
    # Process and move to gpu
    inputs = processor(batch_images, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        all_features.append(features.cpu())

# Concatenate all the (batch_size,1024) 
final_tensor = torch.cat(all_features, dim=0)

#(13652, 1024)
print(f"Final extracted features shape: {final_tensor.shape}")

torch.save(final_tensor, "phikon_features_all.pt")