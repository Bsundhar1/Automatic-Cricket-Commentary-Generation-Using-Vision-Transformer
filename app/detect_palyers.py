from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def is_standing(image, bbox):
    """
    Uses MediaPipe to check if a detected person is standing or seated.
    :param image: PIL image.
    :param bbox: Bounding box (x1, y1, x2, y2) from YOLO.
    :return: True if the person is standing, False if sitting (likely a spectator).
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Crop the detected person from the frame
    person_crop = image_cv[y1:y2, x1:x2]
    if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
        return False  # Ignore invalid detections

    # Convert to RGB for MediaPipe
    person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    
    # Run pose estimation
    results = pose.process(person_crop_rgb)
    
    if results.pose_landmarks:
        # Check knee and hip position to determine if standing
        landmarks = results.pose_landmarks.landmark
        if landmarks[mp_pose.PoseLandmark.LEFT_HIP].y < landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y:
            return True  # Likely a player (standing)
    
    return False  # Likely a spectator (sitting)

def detect_players(image_path):
    """
    Detect players in a given image using YOLO, filtering out spectators using pose estimation.
    :param image_path: Path to the image file.
    :return: Bounding boxes of detected players and count.
    """
    image = Image.open(image_path).convert("RGB")  # Load image
    results = model(image)  # Run YOLO detection

    players = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Get class ID
            if cls == 0:  # Class '0' is 'person'
                bbox = box.xyxy[0].tolist()
                if is_standing(image, bbox):  # Check if the person is standing
                    players.append(bbox)

    return players, len(players)  # Return only standing players
