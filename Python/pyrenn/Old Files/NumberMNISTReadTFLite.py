import tflite_runtime.interpreter as tflite
import numpy as np
import gzip

f = gzip.open('./datasets/numberMNIST/t10k-images-idx3-ubyte.gz', 'r')
image_size = 28
num_images = 10_000
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size)
f.close()

f = gzip.open('./datasets/numberMNIST/t10k-labels-idx1-ubyte.gz', 'r')
f.read(8)
test_labels = []
for i in range(0, 10_000):
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    test_labels.append(labels)
f.close()

class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
path_model = "./SavedNN/NumberMNIST/mnist_model.tflite"

# Data Preprocessing
test_images = data / 255

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=path_model,
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
for index in range(10_000):
    interpreter.set_tensor(input_details[0]['index'], input_data_array[index])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data_array.append(output_data)

'''
index = 111
print("predictions image 1: ", output_data_array[index][0])
print("Highest value predictions image 1: ", np.argmax(output_data_array[index][0]))
index2 = test_labels[index]
print("Corresponding label prediction image 1: ", class_names[index2[0]])
'''
