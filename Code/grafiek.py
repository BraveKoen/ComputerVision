import os


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pickle


# open a file, where you stored the pickled data
file = open('Model_test_8_0.8865079283714294_history', 'rb')

# dump information to that file
history = pickle.load(file)

# close the file
file.close()

acc = history["accuracy"]
val_acc = history["val_accuracy"]
loss = history["loss"]
val_loss = history["val_loss"]

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