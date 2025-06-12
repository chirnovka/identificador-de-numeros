import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from ipywidgets import interact,fixed, interact_manual, IntSlider
import ipywidgets as widgets

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def show_images():
    array = np.random.randint(low=1, high=10000, size=400)
    fig = plt.figure(figsize=(30, 35))
    for i in range(400):
        fig.add_subplot(20, 20, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(y_train[array[i]], color='red', fontsize=20)
        plt.imshow(x_train[array[i]], cmap="gray")
show_images()

y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)

x_train_norm = x_train / 255.
x_test_norm = x_test / 255.

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.optimizers import SGD

model = Sequential([Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
                    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dense(100, activation='relu', kernel_initializer='he_uniform'),
                    BatchNormalization(),
                    Dense(10, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])
model.summary()

x_train_norm = x_train_norm.reshape((x_train_norm.shape[0], 28, 28, 1))
x_test_norm = x_test_norm.reshape((x_test_norm.shape[0], 28, 28, 1))

from sklearn.model_selection import train_test_split as tts

x_val, x_test_, y_val, y_test_ = tts(x_test_norm, y_test_enc, test_size=0.5)

print(x_train_norm.shape)
print(x_test_norm.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test_.shape)
print(y_test_.shape)

history = model.fit(x=x_train_norm, y=y_train_enc,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    epochs=15)