
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape

y_train = y_train.reshape(-1, )
y_test = y_test.reshape(-1, )

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

X_train = X_train / 255.0
X_test = X_test / 255.0

model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # i have a wide screen, deal with it
model.summary()

model.fit(X_train, y_train, epochs=15, validation_data=[X_test, y_test], shuffle=True)

import matplotlib.pyplot as plt

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

plot_sample(X_test, y_test, 2)

import numpy as np
outputY = model.predict(X_test[:5])
y_classes = [np.argmax(element) for element in outputY]
print(y_test[:5])
print(y_classes)