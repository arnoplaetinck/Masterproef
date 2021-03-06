from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

file = tf.keras.utils.get_file(
  "grace_hopper.jpg",
  "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
  x[tf.newaxis, ...])

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
pretrained_model = tf.keras.applications.MobileNet()


converter = tf.lite.TFLiteConverter.from_saved_model("./tmp/mobilenet/1/")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print(list(tflite_model.signatures.keys()))  # ["serving_default"]

infer = tflite_model.signatures["serving_default"]
print(infer.structured_outputs)


labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]

decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]

print("Result after saving and loading:\n", decoded)


