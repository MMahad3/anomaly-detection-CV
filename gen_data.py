import os
import numpy as np
import cv2

# Paths to dataset folders and target size
train_path = "./dataset/Train"  # Replace with your training dataset folder path
test_path = "./dataset/Test"  
image_size = (64, 64)           


def preprocess_dataset(dataset_path, image_size):
    """Preprocess images and labels from a given dataset path."""
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

    return images, labels, class_names


def preprocess_and_save(train_path, test_path, image_size, save_path):
    """Preprocess train and test datasets and save to a compressed .npz file."""
    # Process training data
    print("Processing training data...")
    train_images, train_labels, class_names = preprocess_dataset(train_path, image_size)
    
    # Process testing data
    print("Processing testing data...")
    test_images, test_labels, _ = preprocess_dataset(test_path, image_size)

    # Map class names to numeric labels
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    train_labels = np.array([class_to_label[label] for label in train_labels])
    test_labels = np.array([class_to_label[label] for label in test_labels])

    # Convert to NumPy arrays and normalize
    X_train = np.array(train_images, dtype=np.float32) / 255.0
    y_train = np.array(train_labels)
    X_test = np.array(test_images, dtype=np.float32) / 255.0
    y_test = np.array(test_labels)

    # Save the preprocessed dataset
    np.savez_compressed(save_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test,
                        class_names=np.array(class_names))
    print("Dataset saved successfully!")


# Specify save path
save_path = "./preprocessed_dataset/dataset.npz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Preprocess and save the dataset
preprocess_and_save(train_path, test_path, image_size, save_path)
