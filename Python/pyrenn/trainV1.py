import time
import pyrenn as prn
from numpy import genfromtxt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pynvml import *
import matplotlib.pyplot as plt
import nvidia_smi

nvmlInit()
print("Driver Version:", nvmlSystemGetDriverVersion())
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("Device", i, ":", nvmlDeviceGetName(handle))
nvmlShutdown()
# Get some knowledge about current environment
print("TensorFlow version {}".format(tf.__version__))
print("Eager mode: ", tf.executing_eagerly())
print("Is GPU available: ", tf.test.is_gpu_available())


nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

cores = []
cpu_percent = []
virtual_mem = []
time_start = []
time_stop = []
time_diff = []
time_total_start = []
time_total_end = []
time_total = 0
iterations = 20
labels = ["compair", "friction", "narendra4", "pt2",
          "P0Y0_narendra4", "P0Y0_compair", "gradient", "Totaal"]

###
# Creating a filename
seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "MP_NN_ALL_TRAIN_PC"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_compair.py
print("example_compair")
# Read Example Data
df = genfromtxt('example_data_compressed_air.csv', delimiter=',')
P = np.array([df[1], df[2], df[3]])
Y = np.array([df[4], df[5]])
Ptest = np.array([df[6], df[7], df[8]])
Ytest = np.array([df[9], df[10]])

# Create and train NN
net = prn.CreateNN([3, 5, 5, 2], dIn=[0], dIntern=[], dOut=[1])
net = prn.train_LM(P, Y, net, k_max=500, E_stop=1e-5)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

###
# Save outputs to certain file
prn.saveNN(net, "./SavedNN/compair.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_friction.py
print("friction")

# Read Example Data
df = genfromtxt('example_data_friction.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]

# Create and train NN
net = prn.CreateNN([1, 3, 3, 1])
net = prn.train_LM(P, Y, net, k_max=100, E_stop=9e-4)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

# Save outputs to certain file
prn.saveNN(net, "./SavedNN/friction.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_narendra4.py
print("narendra4")

# Read Example Data
df = genfromtxt('example_data_narendra4.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]

# Create and train NN
net = prn.CreateNN([1, 3, 3, 1], dIn=[1, 2], dIntern=[], dOut=[1, 2, 3])
net = prn.train_LM(P, Y, net, k_max=200, E_stop=1e-3)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
###
# Save outputs to certain file
prn.saveNN(net, "./SavedNN/narendra4.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_pt2.py
print("pt2")

# Read Example Data
df = genfromtxt('example_data_friction.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]

# Create and train NN
net = prn.CreateNN([1, 2, 2, 1], dIn=[0], dIntern=[1], dOut=[1, 2])
net = prn.train_LM(P, Y, net, k_max=100, E_stop=1e-3)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
###
# Save outputs to certain file
prn.saveNN(net, "./SavedNN/pt2.csv")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_using_P0Y0_narendra4.py
print("P0Y0_narendra4")

# Read Example Data
df = genfromtxt('example_data_narendra4.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest_ = df[3]
Ytest_ = df[4]

# define the first 3 timesteps t=[0,1,2] of Test Data as previous (known) data P0test and Y0test
P0test = Ptest_[0:3]
Y0test = Ytest_[0:3]
# Use the timesteps t = [3..99] as Test Data
Ptest = Ptest_[3:100]
Ytest = Ytest_[3:100]

# Create and train NN
net = prn.CreateNN([1, 3, 3, 1], dIn=[1, 2], dIntern=[], dOut=[1, 2, 3])
net = prn.train_LM(P, Y, net, k_max=200, E_stop=1e-3)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
###
# Save outputs to certain file
prn.saveNN(net, "./SavedNN/using_P0Y0_narendra4.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
print("gradient")

df = genfromtxt('example_data_pt2.csv', delimiter=',')
P = df[1]
Y = df[2]

# Create and train NN
net = prn.CreateNN([1, 2, 2, 1], dIn=[0], dIntern=[1], dOut=[1, 2])

# Prepare input Data for gradient calculation
data, net = prn.prepare_data(P, Y, net)

# Calculate derivative vector (gradient vector)
# Real Time Recurrent Learning
t0_rtrl = time.time()
J, E, e = prn.RTRL(net, data)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
g_rtrl = 2 * np.dot(J.transpose(), e)  # calculate g from Jacobian and error vector
t1_rtrl = time.time()

# Back Propagation Through Time
t0_bptt = time.time()
g_bptt, E = prn.BPTT(net, data)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
t1_bptt = time.time()

###
# Save outputs to certain file
prn.saveNN(net, "./SavedNN/gradient.csv")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FashionMNISt.py
print("FashionMNIST")


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
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
# training the model
model.fit(train_images, train_labels, epochs=5)
res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
# Saving model
tf.saved_model.save(model, path_model)
'''
# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Saving tflite model
open(path_model + "fashionMNISTmodel.tflite", "wb").write(tflite_quant_model)
'''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NumberMNIST.py
print("NumberMNIST")


mnist = tf.keras.datasets.mnist
(images_train, labels_train), (images_test, labels_test) = mnist.load_data()
class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

print("Data type:", type(images_train))
print("Dataset shape:", images_train.shape)

print("Labels:", len(labels_train))
print("Possible values:", np.unique(labels_train))

plt.figure()
plt.imshow(images_train[0])
plt.colorbar()
plt.grid(False)
plt.xlabel("Classification label: {}".format(labels_train[0]))
# plt.show()

images_train = images_train / 255.0
images_test = images_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

model.fit(images_train, labels_train, epochs=5)

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

test_loss, test_acc = model.evaluate(images_test, labels_test)
print('Test accuracy:', test_acc)

example_img = images_test[0]

plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(example_img, cmap=plt.cm.binary)


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


predictions = model.predict(images_test)

# plt.show()

model.summary()
loss, acc = model.evaluate(images_test, labels_test)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
print("Restored model, loss: {}".format(loss))

model_path = "./SavedNN/NumberMNIST/"

model.save(model_path + "NumberMNISTtest.h5")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py