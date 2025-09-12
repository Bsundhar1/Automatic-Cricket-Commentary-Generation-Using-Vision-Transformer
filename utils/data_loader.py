import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_data(image_folder, commentary_file):
    images = []
    commentaries = []
    
    with open(commentary_file, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        img_name, commentary = line.strip().split('\t')
        img_path = os.path.join(image_folder, img_name.split('#')[0])
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        images.append(img)
        commentaries.append(commentary)
    
    return np.array(images), np.array(commentaries)