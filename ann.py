import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt



# Load dataset
data = np.load("./preprocessed_dataset/dataset.npz")

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
class_names = data['class_names']

print(f"Loaded dataset:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
# print(y_train[:5])
# print(class_names[5])

# Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,  # Reduced rotation range
#     width_shift_range=0.1,  # Reduced width shift
#     height_shift_range=0.1,  # Reduced height shift
#     shear_range=0.1,  # Reduced shear
#     zoom_range=0.1,  # Reduced zoom range
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# datagen.fit(X_train)

def plot_sample(X,y,index):
    plt.figure(figsize=(3,2))
    plt.imshow(X[index])
    plt.xlabel(class_names[y[index]])
    plt.show()

ann = models.Sequential([
    Input(shape=(64,64,3)),
    layers.Flatten(),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(14, activation='sigmoid')
])

ann.compile(optimizer='SGD',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# ann.fit(datagen.flow(X_train,y_train, batch_size=128),epochs=5)
ann.fit(X_train,y_train,epochs=5)
ann.evaluate(X_test,y_test)

y_pred= ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
# print("classification_report: \n",classification_report(y_test,y_pred_classes,target_names=class_names, zero_division=0))
print("classification_report\n",classification_report(y_test,y_pred_classes))