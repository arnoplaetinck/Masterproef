import tensorflow as tf
from tensorflow import keras
import numpy as np


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Preprocessing
train_images = train_images / 255
test_images = test_images / 255

# Building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax") # for last layer to have probability for each given class (total =1)
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# training the model
model.fit(train_images, train_labels, epochs=5)

# evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

# Making predictions
predictions = model.predict(test_images)
print("predictions image 1: ", predictions[0])
print("Highest value predictions image 1: ", np.argmax(predictions[0]))
print("Corresponding label prediction image 1: ", class_names[test_labels[0]])

