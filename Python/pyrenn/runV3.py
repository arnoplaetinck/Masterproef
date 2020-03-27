import numpy as np
from numpy import genfromtxt
import pyrenn as prn
import csv
import time
from statistics import mean
import psutil
from PIL import Image
import tensorflow as tf
import numpy as np
import threading
# from tensorflow import Session
import gzip

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
          "P0Y0_narendra4", "P0Y0_compair", "gradient", "Im Rec", "FashionMNIST", "Totaal"]

###
# Creating a filename

seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "MP_NN_ALL_RUN_PC"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')


class myThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        cores.append(psutil.cpu_percent(interval=0.05, percpu=True))

    def run2(self):
        cores.append(psutil.cpu_percent(interval=6, percpu=True))

    def run3(self):
        cores.append(psutil.cpu_percent(interval=0.7, percpu=True))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# first time calling cpu percent to get rid of 0,0
thread1 = myThread("Thread-1")
thread1.start()

psutil.cpu_percent(interval=0.1, percpu=True)
time_total_start = time.time()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_compair.py
# This is an example of a dynamic system with 2 outputs and 3 inputs
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_compressed_air.csv', delimiter=',')
    df = df.transpose()

    P = np.array([df[1][1:-1], df[2][1:-1], df[3][1:-1]])
    Y = np.array([df[4][1:-1], df[5][1:-1]])
    Ptest = np.array([df[6][1:-1], df[7][1:-1], df[8][1:-1]])
    Ytest = np.array([df[9][1:-1], df[10][1:-1]])

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/compair.csv")

    thread1.run()
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())

    thread1.join()
print(cores)
print("Done")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_friction.py
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/friction.csv")

    thread1.run()
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_narendra4.py
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_narendra4.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/narendra4.csv")

    thread1.run()
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_pt2.py
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/pt2.csv")

    thread1.run()
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_using_P0Y0_narendra4.py
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_narendra4.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # define the first 3 time steps t=[0,1,2] of Test Data as previous (known) data P0test and Y0test
    P0test = Ptest[0:3]
    Y0test = Ytest[0:3]
    # Use the time steps t = [3..99] as Test Data
    Ptest = Ptest[3:100]
    Ytest = Ytest[3:100]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/using_P0Y0_narendra4.csv")

    thread1.run()
    time_start.append(time.time())

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example__using_P0Y0_compair.py
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_compressed_air.csv', delimiter=',')
    df = df.transpose()

    P = np.array([df[1][1:-1], df[2][1:-1], df[3][1:-1]])
    Y = np.array([df[4][1:-1], df[5][1:-1]])
    Ptest = np.array([df[6][1:-1], df[7][1:-1], df[8][1:-1]])
    Ytest = np.array([df[9][1:-1], df[10][1:-1]])

    # define the first timestep t=0 of Test Data as previous (known) data P0test and Y0test
    P0test = Ptest[:, 0:1]
    Y0test = Ytest[:, 0:1]
    # Use the timesteps t = [1..99] as Test Data
    Ptest = Ptest[:, 1:100]
    Ytest = Ytest[:, 1:100]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/using_P0Y0_compair.csv")

    thread1.run()
    time_start.append(time.time())

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
print("example_gradient")

for i in range(iterations):
    df = genfromtxt('example_data_pt2.csv', delimiter=',')

    P = df[1]
    Y = df[2]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/gradient.csv")

    thread1.run()
    time_start.append(time.time())

    # Prepare input Data for gradient calculation
    data, net = prn.prepare_data(P, Y, net)

    # Real Time Recurrent Learning
    J, E, e = prn.RTRL(net, data)
    g_rtrl = 2 * np.dot(J.transpose(), e)  # calculate g from Jacobian and error vector

    # Back Propagation Through Time
    g_bptt, E = prn.BPTT(net, data)

    # Compare
    # print('\n\n\nComparing Methods:')
    # print('Time RTRL: ', (t1_rtrl - t0_rtrl), 's')
    # print('Time BPTT: ', (t1_bptt - t0_bptt), 's')
    # if not np.any(np.abs(g_rtrl - g_bptt) > 1e-9):
    #    print('\nBoth methods showing the same result!')
    #    print('g_rtrl/g_bptt = ', g_rtrl / g_bptt)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ImRecKerasRead.py
# 124 images from COCO val2017 dataset
print("ImRecKerasRead")

path_image = "./images/KerasRead/"
extention = ".jpg"

imagenet_labels = np.array(open(path_image + "ImageNetLabels.txt").read().splitlines())

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./SavedNN/ImRecKerasModel/ImRecKerasModel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

input_data2 = []
with open("./images/KerasRead/labels.txt", mode='r') as label_file:
    for label in label_file:
        path = path_image + label.rstrip() + extention
        img = Image.open(path).resize((width, height))
        input_data = np.expand_dims(img, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        input_data2.append(input_data)

for i in range(iterations):
    thread1.run2()
    time_start.append(time.time())

    for data in input_data2:
        interpreter.set_tensor(input_details[0]['index'], data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # decoded = imagenet_labels[np.argsort(output_data)[0, ::-1][:5] + 1]
        # print("Result AFTER saving: ", decoded)

    time_stop.append(time.time())
    thread1.join()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FashionMNISTREAD.py
print("Fashion MNISTREAD")

f = gzip.open('./datasets/fashionMNIST/t10k-images-idx3-ubyte.gz', 'r')
image_size = 28
num_images = 10_000
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size)
f.close()

f = gzip.open('./datasets/fashionMNIST/t10k-labels-idx1-ubyte.gz', 'r')
f.read(8)
test_labels = []
for i in range(0, 10_000):
    buf = f.read(1)
    inhoud = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    test_labels.append(inhoud)
f.close()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
path_model = "./SavedNN/FashionMNIST/fashionMNISTmodel.tflite"

# Data Preprocessing
test_images = data / 255

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=path_model)
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
for i in range(iterations):
    output_data_array = []

    thread1.run3()
    time_start.append(time.time())

    for index in range(10000):
        interpreter.set_tensor(input_details[0]['index'], input_data_array[index])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data_array.append(output_data)

    time_stop.append(time.time())
    thread1.join()

time_total_end = time.time()
cores.append(psutil.cpu_percent(interval=2, percpu=True))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Logging data
for i in range(iterations * (len(labels) - 1)):
    time_diff.append(time_stop[i] - time_start[i])
    time_total += time_stop[i] - time_start[i]
time_diff.append(time_total / iterations)
i = 0
for core in cores:
    cpu_percent.append(mean(cores[i]))
    i += 1
i = 0

with open('./logging/' + naam + ".csv", mode='w') as results_file:
    fieldnames = ['Naam', 'CPU Percentage', 'timediff', 'virtual mem']
    file_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    file_writer.writeheader()
    for i in range(iterations * (len(labels) - 1) + 1):
        j = int(i / iterations)
        file_writer.writerow({'Naam': labels[j], 'CPU Percentage': str(cpu_percent[i]), 'timediff': str(time_diff[i])
                              })
