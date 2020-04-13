import matplotlib.pyplot as plt
import tensorflow as tf
import keras
#

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7897  # You may select anything up to 60,000
print(y_train[image_index])  # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
# plt.show()

print("x_train.shape: ", x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
'''model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
'''

# Compiling and fitting the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=1)

# Evaluating the model
model.evaluate(x_test, y_test)

# Individual prediction
image_index = 4337
img_rows = 28
img_cols = 28
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
# plt.show()
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())

path_model = "./SavedNN/NumberMNIST/"
model.save(path_model + "NumberMNISTtest.h5")


new_model = keras.models.load_model(path_model + "NumberMNISTtest.h5")

# Converting to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Saving tflite model
open(path_model + "NumberMNISTtest.tflite", "wb").write(tflite_model)
