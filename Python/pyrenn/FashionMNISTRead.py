import tensorflow as tf
from tensorflow import keras
import numpy as np


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
path_model = "./SavedNN/FashionMNIST/fashionMNISTmodel.tflite"

# Data Preprocessing
test_images = test_images / 255
print("test_images: ", test_images.shape)
print("test_images: ", test_images[125].shape)
print("test_images: ", np.argmax(test_images))

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=path_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("input_details: ", input_details)
# print("output_details: ", output_details)


# Get input and output shapes
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Test model on  input data.
input_shape = input_details[0]['shape']

# Preprocessing data 2
index = 0
input_data_array = []
for image in test_images:
    input_data = np.expand_dims(test_images[index], axis=0)
    index += 1
    if floating_model:
        input_data = np.float32(input_data)
    input_data_array.append(input_data)

output_data_array = []
for index in range(10000):
    interpreter.set_tensor(input_details[0]['index'], input_data_array[index])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data_array.append(output_data)

index = 536
print("predictions image 1: ", output_data_array[index][0])
print("Highest value predictions image 1: ", np.argmax(output_data_array[index][0]))
print("Corresponding label prediction image 1: ", class_names[test_labels[index]])
