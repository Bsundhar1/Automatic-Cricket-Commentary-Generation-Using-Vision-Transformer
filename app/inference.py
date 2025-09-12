import torch
import pickle
import numpy as np
import random
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity

# Load saved features & commentaries
with open("/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/model/image_features.pkl", "rb") as f:
    image_features = pickle.load(f)
with open("/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/model/commentaries.pkl", "rb") as f:
    commentaries = pickle.load(f)

# Load ViT Model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Extract features from a new image
def extract_features(image):
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vision_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Match input image to the closest stored image
def find_best_match(uploaded_image):
    uploaded_features = extract_features(uploaded_image)

    best_match = None
    best_score = -1
    for img_path, features in image_features.items():
        similarity = cosine_similarity(uploaded_features.numpy(), features.numpy())[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = img_path

    matched_commentaries = commentaries.get(best_match, ["No commentary found."])
    return best_match, random.choice(matched_commentaries)

# Test with an image
if __name__ == "__main__":
    test_image_path = "/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/data/images/sample.png"
    test_image = Image.open(test_image_path)
    match, commentary = find_best_match(test_image)
    print(f"📢 Matched Commentary: {commentary}")
