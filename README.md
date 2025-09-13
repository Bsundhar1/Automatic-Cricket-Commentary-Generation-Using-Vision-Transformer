# 🏏Automatic Cricket Commentary Generation using Vision Transformer

This project is an **AI-powered cricket commentary system** that generates real-time match commentary from video input using **Vision Transformers (ViT), YOLO, GPT, and Streamlit**.  
It detects cricketing events (like Four, Six, Bowled) from video frames and produces **natural language commentary + audio output**.

---

## 🚀 Features
- 🎥 **Event Detection**: Detects cricket shots (Four, Six, Bowled, etc.) using **YOLO + Vision Transformers**  
- 📝 **Commentary Generation**: Uses GPT-based text generation for natural commentary  
- 🔊 **Audio Output**: Converts generated text to speech for live-like experience  
- 📊 **Evaluation**: Supports precision, recall, and F1-score evaluation against ground truth  
- 🌐 **Interactive UI**: Built with **Streamlit** for easy video upload and live commentary display  

---

## 🛠️ Tech Stack
- **Programming Language**: Python  
- **Frontend/UI**: Streamlit  
- **Deep Learning**: Vision Transformer (ViT), YOLO, GPT  
- **NLP**: GPT-based text generation, duplicate filtering  
- **TTS**: gTTS (Google Text-to-Speech)  
- **Data Handling**: NumPy, Pandas  
- **Model Training**: TensorFlow / PyTorch  

---

## 📂 Project Structure
cricomm/
│── app/ # Main application scripts
│ ├── app.py # Entry point for Streamlit app
│ ├── train.py # Model training script
│ ├── inference.py # Event + commentary inference
│ └── detect_players.py
│
│── utils/ # Utility scripts
│ ├── data_loader.py
│ └── preprocess.py
│
│── data/ (⚠️ Not included in repo - download separately)
│ ├── Four/ # Images for "Four" events
│ ├── bowled/ # Images for "Bowled" events
│ ├── commentary.txt # Ground truth commentary
│ └── model/ # Trained models
│
│── model/ (⚠️ Not included - download separately)
│ ├── image_features.pkl
│ ├── commentaries.pkl
│ └── model.h5
│
│── worksheet.ipynb # Notebook for experiments
│── overallcom.txt # Combined commentary dataset


---

## 📦 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Automatic-Cricket-Commentary-Generation-Using-Vision-Transformer.git
   cd Automatic-Cricket-Commentary-Generation-Using-Vision-Transformer

2. pip install -r requirements.txt
3. download datasets yourselves so that you can train a model yourself.


Usage

Run the Streamlit app:

streamlit run app/app.py


Upload a cricket video

The system extracts frames, detects events, and generates commentary

Commentary is displayed on screen and converted to speech


📑 Publication

“Automatic Cricket Commentary Generation using Vision Transformers”
Published in [IJRASET], [April 2025].
https://share.google/5T5r2cemTUjG4W2Ed - Research Paper
https://share.google/BwIgznU48PBJIyTfp - ResearchGate Link


👨‍💻 Author
Balasundhar K J
📧 balasunder961@gmail.com
 | 📍 Karaikal, India




