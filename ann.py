import numpy as np
from sklearn.model_selection import train_test_split
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

model_h5_path = "./saved_model/my_ann_model.h5"
model.save(model_h5_path)  # Save in HDF5 format
print(f"\nModel saved in HDF5 format at: {model_h5_path}")

y_pred= ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
# print("classification_report: \n",classification_report(y_test,y_pred_classes,ta
print("classification_report\n",classification_report(y_test,y_pred_classes))
