import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report
import pandas as pd


# Path to dataset folder
dataset_path = "./dataset/Train"  # Replace with your dataset folder path
image_size = (64, 64)  # ResNet input size

# Load images and labels
def load_images_from_folders(dataset_path, image_size):
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

# Load data
images, labels, class_names = load_images_from_folders(dataset_path, image_size)

print("\nData loading done")

# Convert to NumPy arrays
X = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values
y = pd.get_dummies(labels).values  # One-hot encode class labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load Pre-trained ResNet50 Model (without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Add a new classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(y.shape[1], activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nStarted training")

# Custom callback for progress display
class TrainingProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f" - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f} "
              f"- Validation Loss: {logs['val_loss']:.4f} - Validation Accuracy: {logs['val_accuracy']:.4f}")

# Train the model
batch_size = 32
epochs = 10

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=epochs,
    callbacks=[TrainingProgress()],
    verbose=0  # Suppress the default progress bar
)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {scores[1] * 100:.2f}%")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:\n", classification_report(y_true_classes, y_pred_classes, target_names=class_names))
