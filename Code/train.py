import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy
import pickle


rock = os.path.join("./Rock-Paper-Scissors/train/rock/")
paper = os.path.join("./Rock-Paper-Scissors/train/paper/")
scissors = os.path.join("./Rock-Paper-Scissors/train/scissors/")

print("Rock : ",len(os.listdir(rock)))
print("Paper : ",len(os.listdir(rock)))
print("Scissors : ",len(os.listdir(rock)))

rockFiles = os.listdir(rock)
paperFiles = os.listdir(paper)
scissorsFiles = os.listdir(scissors)

model = keras.Sequential([
     keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
     keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),

    
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.summary()
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

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
                   validation_steps=3,epochs=20,
                   steps_per_epoch=20,
                   verbose=1)

name = "Model_test_8_" + str(history.history["accuracy"][-1])

model.save(name)
name = name + "_history"

with open(name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)