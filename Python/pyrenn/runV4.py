from numpy import genfromtxt
import pyrenn as prn
import csv
import time
from statistics import mean
import psutil
from PIL import Image
import tensorflow as tf
import numpy as np
import gzip

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

cores = []
cpu_percent = []
virtual_mem = []
time_start = []
time_stop = []

time_total = 0
iterations = 2
labels = ["compair", "friction", "narendra4", "pt2", "P0Y0_narendra4", "P0Y0_compair", "gradient",
          "FashionMNIST", "NumberMNIST", "catsVSdogs", "Im Rec", "Totaal"]

###
# Creating a filename
seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "Benchmark_PC"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')

with open('./logging/' + naam + ".csv", mode='w') as results_file:
    fieldnames = ['Naam', 'CPU Percentage', 'timediff']
    file_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    file_writer.writeheader()


def logging_data(program_index, stop, start, cpu):
    # Logging data
    cores_avg = mean(cpu)
    time_diff = stop-start

    with open('./logging/' + naam + ".csv", mode='a+') as data_file:
        data_writer = csv.DictWriter(data_file, fieldnames=fieldnames)
        data_writer.writerow(
            {'Naam': labels[program_index],
             'CPU Percentage': str(cores_avg),
             'timediff': str(time_diff)
             })


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# first time calling cpu percent to get rid of 0,0
psutil.cpu_percent(interval=None, percpu=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_compair.py
# This is an example of a dynamic system with 2 outputs and 3 inputs
print("compair")
iteration = 0
while iteration < iterations:
    # Read Example Data
    df = genfromtxt('example_data_compressed_air.csv', delimiter=',')
    df = df.transpose()

    P = np.array([df[1][1:-1], df[2][1:-1], df[3][1:-1]])
    Y = np.array([df[4][1:-1], df[5][1:-1]])
    Ptest = np.array([df[6][1:-1], df[7][1:-1], df[8][1:-1]])
    Ytest = np.array([df[9][1:-1], df[10][1:-1]])

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/compair.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(0, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_friction.py
print("friction")
iteration = 0
while iteration < iterations:    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/friction.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(1, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_narendra4.py
print("narendra4")
iteration = 0
while iteration < iterations:
    # Read Example Data
    df = genfromtxt('example_data_narendra4.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/narendra4.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(2, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_pt2.py
print("pt2")
iteration = 0
while iteration < iterations:
    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/pt2.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(3, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_using_P0Y0_narendra4.py
print("using_P0Y0_narendra4")
iteration = 0
while iteration < iterations:
    # Read Example Data
    df = genfromtxt('example_data_narendra4.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]
    Ptest = df[3][1:-1]
    Ytest = df[4][1:-1]

    # define the first 3 timesteps t=[0,1,2] of Test Data as previous (known) data P0test and Y0test
    P0test = Ptest[0:3]
    Y0test = Ytest[0:3]
    # Use the timesteps t = [3..99] as Test Data
    Ptest = Ptest[3:100]
    Ytest = Ytest[3:100]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/using_P0Y0_narendra4.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(4, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
print("gradient")

iteration = 0
while iteration < iterations:
    df = genfromtxt('example_data_pt2.csv', delimiter=',')
    df = df.transpose()

    P = df[1][1:-1]
    Y = df[2][1:-1]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/gradient.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    # Prepare input Data for gradient calculation
    data, net = prn.prepare_data(P, Y, net)

    # Real Time Recurrent Learning
    J, E, e = prn.RTRL(net, data)
    g_rtrl = 2 * np.dot(J.transpose(), e)  # calculate g from Jacobian and error vector

    # Back Propagation Through Time
    # g_bptt, E = prn.BPTT(net, data)

    # Compare
    # print('\n\n\nComparing Methods:')
    # print('Time RTRL: ', (t1_rtrl - t0_rtrl), 's')
    # print('Time BPTT: ', (t1_bptt - t0_bptt), 's')
    # if not np.any(np.abs(g_rtrl - g_bptt) > 1e-9):
    #    print('\nBoth methods showing the same result!')
    #    print('g_rtrl/g_bptt = ', g_rtrl / g_bptt)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(6, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)

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

iteration = 0
while iteration < iterations:
    output_data_array = []

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    for index in range(10000):
        interpreter.set_tensor(input_details[0]['index'], input_data_array[index])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data_array.append(output_data)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(7, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# NumberMNISTREAD.py
print("NumberMNISTREAD")

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
    label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    test_labels.append(label)
f.close()

class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
path_model = "./SavedNN/NumberMNIST/mnist_model.tflite"

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

iteration = 0
while iteration < iterations:
    output_data_array = []

    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    for index in range(10_000):
        interpreter.set_tensor(input_details[0]['index'], input_data_array[index])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data_array.append(output_data)

    time_stop = time.time()
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(8, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# catsVSdogs10Read.py
# 124 images from COCO val2017 dataset
print("catsVSdogs10Read")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
path_image = "./images/catsVSdogs/"
extention = ".jpg"
path_model = "./SavedNN/catsVSdogs/"
labels_array = np.arange(0, 39)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=path_model + "catsVSdogsmodel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on  input data.
input_shape = input_details[0]['shape']
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print(height, width)
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
    # img.show()
    img.save("./images/test/" + str(label) + ".png")
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data)) / 255
    input_data2.append(input_data)


iteration = 0
while iteration < iterations:
    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    for data in input_data2:
        interpreter.set_tensor(input_details[0]['index'], data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        npa = np.asarray(output_data[0], dtype=np.float32)
        # print(class_names[np.argmax(npa)])

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(9, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ImRecKerasRead.py
# 124 images from COCO val2017 dataset
print("ImRecKerasRead")

path_image = "./images/KerasRead/"
extention = ".jpg"

imagenet_labels = np.array(open(path_image+"ImageNetLabels.txt").read().splitlines())

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./SavedNN/ImRecKerasModel/ImRecKerasModel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print("height: ", height, " width", width)

input_data2 = []
with open("./images/KerasRead/labels.txt", mode='r') as label_file:
    for label in label_file:
        path = path_image + label.rstrip() + extention
        img = Image.open(path).resize((width, height))
        input_data = np.expand_dims(img, axis=0)
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        input_data2.append(input_data)

iteration = 0
while iteration < iterations:
    psutil.cpu_percent(interval=None, percpu=True)
    time_start = time.time()

    for data in input_data2:
        interpreter.set_tensor(input_details[0]['index'], data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        decoded = imagenet_labels[np.argsort(output_data)[0, ::-1][:5] + 1]
        print("Result AFTER saving: ", decoded)

    time_stop = (time.time())
    cores = psutil.cpu_percent(interval=None, percpu=True)
    if (mean(cores) != 0.0) and (time_stop-time_start != 0):
        logging_data(10, time_stop, time_start, cores)
        iteration += 1
        time_total += time_stop - time_start
    print("iteration: ", iteration, " mean cores: ", mean(cores), " time_stop-time_start: ", time_stop-time_start)

cores = psutil.cpu_percent(interval=2, percpu=True)
with open('./logging/' + naam + ".csv", mode='a+') as data_file:
    data_writer = csv.DictWriter(data_file, fieldnames=fieldnames)

    data_writer.writerow(
        {'Naam': labels[-1],
         'CPU Percentage': str(mean(cores)),
         'timediff': str(time_total/iterations)
         })
