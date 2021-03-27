import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd 

from keras.preprocessing import image
import pickle

model = keras.models.load_model('./trained_model_64_one')


train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("./Rock-Paper-Scissors/train/",
                                                   batch_size=126,
                                                   target_size=(150,150),
                                                   class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory("./Rock-Paper-Scissors/test/",
                                                             batch_size=126,
                                                             target_size=(150,150),
                                                             class_mode="categorical")

history = model.fit(train_generator, validation_data=validation_generator,
                   validation_steps=1,epochs=5,
                   steps_per_epoch=1,
                   verbose=1)



name = "Model_512_" + str(history.history["accuracy"][-1])

model.save(name)
name = name + "_history"

with open(name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epoch = range(len(acc))

plt.plot(epoch, acc, "r", label = "Training Accuracy")
plt.plot(epoch, val_acc, "b", label = "Validation Accuracy")
plt.title("Training And Validation Accuracy")

plt.legend()
plt.figure()

plt.plot(epoch, loss, "r", label = "Training Loss")
plt.plot(epoch, val_loss, "b", label = "Validation Loss")
plt.title("Training And Validation Loss")

plt.legend()
plt.show()