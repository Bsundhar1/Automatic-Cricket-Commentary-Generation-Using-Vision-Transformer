import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model
import pickle

# Load the model & tokenizer
MODEL_PATH = "/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/model/model.h5"  # Ensure correct path
TOKENIZER_PATH = "/content/drive/MyDrive/cricomm-20250215T125451Z-001/cricomm/model/tokenizer.pkl"

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 19  # Ensure it matches the trained model

# Function to extract frames from a video
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

# Function to process an image frame
def extract_features(image):
    img = cv2.resize(image, (224, 224))  # Resize to model input size
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to generate commentary for a frame
def generate_commentary(frame):
    image = extract_features(frame)
    input_text = np.zeros((1, max_len))  # ✅ Ensure correct shape

    commentary_words = []
    for i in range(max_len - 1):
        preds = model.predict([image, input_text])  # ✅ Ensure it's a list input
        next_word = np.argmax(preds[0, i, :])

        word = tokenizer.index_word.get(next_word, '')
        if word == '<end>':
            break
        
        commentary_words.append(word)
        input_text[0, i] = next_word  # ✅ Append predicted word

    commentary = ' '.join(commentary_words)
    return commentary.strip()

# Streamlit UI
st.title("🏏 AI-Powered Cricket Commentary Generator")
st.write("Upload a **video** or a **single image** to generate AI commentary!")

# User chooses between Video or Image
option = st.radio("Select Input Type:", ["Upload Video", "Upload Image"])

if option == "Upload Video":
    video_file = st.file_uploader("Upload a cricket video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if video_file:
        st.video(video_file)  # Display uploaded video
        
        # Save the uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.write("🔄 Extracting frames...")

        # Extract frames & generate commentary
        frames = extract_frames(video_path)
        commentary_list = [generate_commentary(frame) for frame in frames]

        st.write("📢 **Generated Commentary:**")
        for idx, comment in enumerate(commentary_list):
            st.write(f"🔹 Frame {idx + 1}: {comment}")

        # Cleanup
        os.remove(video_path)

elif option == "Upload Image":
    image_file = st.file_uploader("Upload a single image frame", type=["jpg", "jpeg", "png"])

    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        # Read the uploaded image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Generate commentary
        commentary = generate_commentary(frame)

        st.write("📢 **Generated Commentary:**")
        st.write(f"🗣 {commentary}")
