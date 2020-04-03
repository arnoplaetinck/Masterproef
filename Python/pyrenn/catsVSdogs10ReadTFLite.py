# Imports
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
path_image = "./images/catsVSdogs/"
extention = ".jpg"
path_model = "./SavedNN/catsVSdogs/"
labels_array = np.arange(0, 39)

# Data preprocessing
IMG_SIZE = 160  # All images will be resized to 160x160


# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=path_model + "catsVSdogsmodel.tflite",
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on  input data.
input_shape = input_details[0]['shape']
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
input_data2 = []

for label in labels_array:
    path = path_image + "/Cat/" + str(label) + extention
    img = Image.open(path).resize((width, height))
    input_data = np.expand_dims(img, axis=0)
    if floating_model:
        input_data = (np.float32(input_data)) / 255
    input_data2.append(input_data)
for label in labels_array:
    path = path_image + "/Dog/" + str(label) + extention
    img = Image.open(path).resize((width, height))
    img.show()
    input_data = np.expand_dims(img, axis=0)
    if floating_model:
        input_data = (np.float32(input_data)) / 255
    input_data2.append(input_data)

for data in input_data2:
    interpreter.set_tensor(input_details[0]['index'], data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    npa = np.asarray(output_data[0], dtype=np.float32)
    # print(class_names[np.argmax(npa)])
