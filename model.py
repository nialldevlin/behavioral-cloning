print("Loading Libraries...")

import csv
import cv2
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, Cropping2D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


###Hyperparameters
learn_rate = 0.001
epochs = 10
dropout = 0.5
val_split = 0.2
batch_size = 1500

print("Loading dataset...")
with open('x_train.npy', 'rb') as f:
    x_train = np.load(f)
with open('y_train.npy', 'rb') as f:
    y_train = np.load(f)
print("Dataset loaded")
print("Dataset Length: " + str(len(x_train)))

img_shape = x_train[0].shape
crop_t = 40
crop_b = 20
img_shape_c = (img_shape[0] - (crop_t + crop_b), img_shape[1], img_shape[2])
print(img_shape_c)

model = Sequential()
model.add(Cropping2D(cropping=((crop_t,crop_b), (0,0))))
model.add(Lambda(lambda x: x/127.5-1, input_shape=img_shape_c))
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(50, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mse')
model.fit(
    x_train, 
    y_train, 
    validation_split=val_split, 
    shuffle=True, epochs=epochs, 
    batch_size=batch_size, 
    callbacks=EarlyStopping(patience=2, monitor="val_loss"),
    verbose=1)
model.save("model.h5")
print("Done")