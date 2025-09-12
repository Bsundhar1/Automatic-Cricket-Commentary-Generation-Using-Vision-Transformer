# train.py

import os
import torch
import pickle
from PIL import Image
from collections import defaultdict
from transformers import ViTFeatureExtractor, ViTModel

# === CONFIGURATION ===
event_type = "Four"  # Change this to 'Six', 'Bowled', etc., when training other events
image_folder = f"/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/data/{event_type}"
commentary_file = f"/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/data/{event_type}_commentary.txt"
save_folder = "/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/model"
# === LOAD MODELS ===
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# === UNIFIED FILE PATHS ===
combined_features_path = os.path.join(save_folder, "image_features.pkl")
combined_commentaries_path = os.path.join(save_folder, "commentaries.pkl")

# === LOAD EXISTING DATA (IF ANY) ===
if os.path.exists(combined_features_path):
    with open(combined_features_path, "rb") as f:
        image_features = pickle.load(f)
else:
    image_features = {}

if os.path.exists(combined_commentaries_path):
    with open(combined_commentaries_path, "rb") as f:
        commentaries = pickle.load(f)
else:
    commentaries = defaultdict(list)

# === FEATURE EXTRACTION FUNCTION ===
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vision_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# === LOAD COMMENTARY DATA ===
with open(commentary_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split("\t")
    if len(parts) != 2:
        continue

    img_comment_id, commentary = parts
    img_name = img_comment_id.split("#")[0]
    img_path = os.path.join(image_folder, img_name)

    if not os.path.exists(img_path):
        continue

    if img_path not in image_features:
        image_features[img_path] = extract_features(img_path)

    commentaries[img_path].append(commentary)

# === SAVE UNIFIED DATA ===
os.makedirs(save_folder, exist_ok=True)
with open(combined_features_path, "wb") as f:
    pickle.dump(image_features, f)
with open(combined_commentaries_path, "wb") as f:
    pickle.dump(commentaries, f)

print(f"✅ Training complete for event: {event_type}. Features and commentaries updated.")
