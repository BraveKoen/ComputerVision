import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd 

from keras.preprocessing import image

model = keras.models.load_model('./trained_model')

path = "./Rock-Paper-Scissors/Rock-Paper-Scissors/test/paper/testpaper04-28.png"

img_show = mpimg.imread(path)
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

for i in classes:
    if i[0] == 1:
        print("paper")
    elif i[1] == 1:
        print("Rock")
    else:
        print("Scissor")
        
plt.imshow(img_show)
plt.show()