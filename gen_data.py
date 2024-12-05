import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Dataset path and target size
dataset_path = "./dataset/Train"  # Replace with your dataset folder path
image_size = (64, 64)

def preprocess_and_save(dataset_path, image_size, save_path, test_size=0.2, random_state=42):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))  # Get folder names (class labels)

    for class_name in class_names:
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):  # Ensure it's a folder
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)  # Resize to target size
                    images.append(img)
                    labels.append(class_name)  # Class label from folder name
    
    # Convert labels to numerical values
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([class_to_label[label] for label in labels])

    # Convert to arrays
    X = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values
    y = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save the data in a single file
    np.savez_compressed(save_path, 
                        X_train=X_train, X_test=X_test, 
                        y_train=y_train, y_test=y_test, 
                        class_names=np.array(class_names))
    print("Dataset saved successfully!")

# Specify save path
save_path = "./preprocessed_dataset/dataset.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
preprocess_and_save(dataset_path, image_size, save_path)
