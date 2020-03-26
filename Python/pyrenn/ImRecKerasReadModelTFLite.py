from __future__ import absolute_import, division, print_function, unicode_literals
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# 124 images from COCO val2017 dataset
path_image = "./images/KerasRead/"
extention = ".jpg"

imagenet_labels = np.array(open(path_image+"ImageNetLabels.txt").read().splitlines())

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="./SavedNN/ImRecKerasModel/ImRecKerasModel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on  input data.
input_shape = input_details[0]['shape']

floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

img_set = []
input_data2 = []
with open("./images/KerasRead/labels.txt", mode='r') as label_file:
    for label in label_file:
        path = path_image + label.rstrip() + extention
        img = Image.open(path).resize((width, height))
        input_data = np.expand_dims(img, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # decoded = imagenet_labels[np.argsort(output_data)[0, ::-1][:5] + 1]
        # print("Result AFTER saving: ", decoded)





