import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
'''
with open("./datasets/fashionMNIST/test_images.txt", mode='w') as test_images_file:
    for image in test_images:
        for row in image:
            test_images_file.write(image[row])

with open("./datasets/fashionMNIST/test_labels.txt", mode='w') as test_labels_file:
    for label in test_labels:
        test_labels_file.write(label)
'''
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
path_model = "./SavedNN/FashionMNIST/"

# Data Preprocessing
train_images = train_images / 255
test_images = test_images / 255

# Building the model
model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    # for last layer to have the probability for each given class (total =1)
    keras.layers.Dense(10, activation="softmax")])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# training the model
model.fit(train_images, train_labels, epochs=5)

# Saving model
tf.saved_model.save(model, path_model)

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Saving tflite model
open(path_model + "fashionMNISTmodel.tflite", "wb").write(tflite_quant_model)

# evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

# Making predictions
predictions = model.predict(test_images)
print("predictions image 1: ", predictions[0])
print("Highest value predictions image 1: ", np.argmax(predictions[0]))
print("Corresponding label prediction image 1: ", class_names[test_labels[0]])

