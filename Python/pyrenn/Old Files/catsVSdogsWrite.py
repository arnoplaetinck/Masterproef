# Imports
import os
import numpy as np
import tensorflow as tf

# https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=true&scrollTo=tcoKn1VUieqx
# Using a Pretrained model from google colab
keras = tf.keras


path_model = "./SavedNN/catsVSdogs/"
model = keras.models.load_model(path_model + "catsVSdogs.h5")

# Converting to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Saving tflite model
open(path_model + "catsVSdogs.tflite", "wb").write(tflite_model)


