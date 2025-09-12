import os
import cv2
import torch
import pickle
import numpy as np
import tempfile
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ViTFeatureExtractor, ViTModel
from gtts import gTTS
from sklearn.metrics import precision_score, recall_score, f1_score
import random

# === MODEL PATHS ===
model_dir = "/content/drive/MyDrive/cricomm/model"
feature_file = os.path.join(model_dir, "image_features.pkl")
commentary_file = os.path.join(model_dir, "commentaries.pkl")

# === LOAD FEATURES AND COMMENTARIES ===
with open(feature_file, "rb") as f:
    image_features = pickle.load(f)
with open(commentary_file, "rb") as f:
    commentaries = pickle.load(f)

# === LOAD VIT MODELS ===
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# === FEATURE EXTRACTION ===
def extract_features(image):
    image = image.convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vision_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# === COMMENTARY MATCHING ===
def get_best_match(frame_feature, image_features, commentaries):
    best_score = -1
    best_image_path = None
    for img_path, stored_feature in image_features.items():
        sim = cosine_similarity(frame_feature, stored_feature.numpy())[0][0]
        if sim > best_score:
            best_score = sim
            best_image_path = img_path

    if best_image_path and best_image_path in commentaries:
        options = commentaries[best_image_path]
        return random.choice(options) if options else "No match found"
    return "No match found"

# === VIDEO FRAME EXTRACTOR ===
def extract_keyframes(video_path, interval=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        count += 1
    cap.release()
    return frames

# === METRICS CALCULATION ===
def calculate_metrics(generated, ground_truth_text):
    ground_truth = ground_truth_text.strip().split(" ")
    generated_words = generated.strip().split(" ")
    y_true = [1 if word in ground_truth else 0 for word in generated_words]
    y_pred = [1] * len(generated_words)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

# === AUDIO GENERATION ===
def play_combined_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as speech_file:
        tts.save(speech_file.name)
        st.audio(speech_file.name, format="audio/mp3", autoplay=True)

# === LOAD GROUND TRUTH FROM FILE ===
ground_truth_path = "/content/drive/MyDrive/cricomm-20250215T125451Z-001/overallcom.txt"

if os.path.exists(ground_truth_path):
    with open(ground_truth_path, "r", encoding="utf-8") as gt_file:
        GROUND_TRUTH_COMMENTARY = gt_file.read()
else:
    GROUND_TRUTH_COMMENTARY = ""
    st.warning("⚠️ Ground truth commentary file not found.")


# === STREAMLIT UI ===
st.set_page_config(page_title="CRICOMM - AI Cricket Commentary", layout="wide")
st.title("🏏 CRICOMM - AI-Powered Cricket Commentary")
st.write("Upload a cricket video to generate AI commentary and evaluate performance.")

video_file = st.file_uploader("📺 Upload a cricket video", type=["mp4", "avi", "mov"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_path)
    st.info("⏳ Processing video...")

    frames = extract_keyframes(video_path, interval=60)
    generated_commentaries = []
    seen_commentaries = set()

    for frame in frames:
        frame_feature = extract_features(frame)
        commentary = get_best_match(frame_feature, image_features, commentaries)
        clean_commentary = commentary.strip().replace("\n", " ").rstrip('.') + '.'
        if clean_commentary not in seen_commentaries:
            seen_commentaries.add(clean_commentary)
            generated_commentaries.append(clean_commentary)

    full_commentary = " ".join(generated_commentaries)

    st.success("✅ Commentary generation complete!")
    st.subheader("📢 Generated Commentary Paragraph:")
    st.markdown(full_commentary)

    # === AUDIO PLAYBACK ===
    st.subheader("🔊 AI Commentary Audio")
    play_combined_speech(full_commentary)

    # === METRICS (from internal ground truth) ===
    precision, recall, f1 = calculate_metrics(full_commentary, GROUND_TRUTH_COMMENTARY)
    st.subheader("📊 Evaluation Metrics")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
