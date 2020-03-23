# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=true&scrollTo=tcoKn1VUieqx
# Using a Pretrained model
keras = tf.keras

path_model = "./SavedNN/catsVSdogs/"
# tfds.disable_progress_bar()

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# creates a function object that we can use to get labels
get_label_name = metadata.features['label'].int2str

# Data preprocessing
IMG_SIZE = 160  # All images will be resized to 160x160


def format_example(image, label):
    print("image1: ", image)

    image = tf.cast(image, tf.float32)
    print("image2: ", image)

    # Reshaping values between -1, 1
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    print("image3: ", image)
    return image, label


# resizing images to same length
test = raw_test.map(format_example)

# Creating batches
BATCH_SIZE = 1
SHUFFLE_BUFFER_SIZE = 1000

test_batches = test.batch(BATCH_SIZE)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=path_model + "catsVSdogs.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on  input data.
input_shape = input_details[0]['shape']
print("input_details: ", input_details[0])

floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print("height: ", height, " width: ", width)

'''
if floating_model:
    print("test_batches.element_spec: ", test_batches.element_spec)
    test_batches = np.float32(test_batches)
    test_batches = (test_batches - 127.5) / 127.5
    '''

index_element = 0
for element in test_batches.as_numpy_iterator():
    index_element += 1
    # print("len element: ", len(element))
    # print(" len element[0]: ", len(element[0]))
    # print("shape element: ", np.shape(element))
    # print("element[0]: ", element[0][0])
    # print("element[0]: ", element[1])

    for batch_index in range(BATCH_SIZE):
        interpreter.set_tensor(input_details[0]['index'], element[0])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print("Output_data: ", output_data)

print("index_element: ", index_element)