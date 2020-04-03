from __future__ import absolute_import, division, print_function, unicode_literals
from numpy import genfromtxt
import pyrenn as prn
import csv
import time
from statistics import mean
import psutil
import functools
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

cores = []
cpu_percent = []
virtual_mem = []
time_start = []
time_stop = []
time_diff = []
time_total = 0
iterations = 1
labels = ["compair", "friction", "narendra4", "pt2",
          "P0Y0_narendra4", "P0Y0_compair", "gradient", "Text Classificatie", "Totaal"]

###
# Creating a filename

seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "MP_NN_ALL_RUN_PC"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# first time calling cpu percent to get rid of 0,0
psutil.cpu_percent(interval=None, percpu=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_compair.py
for i in range(iterations):


    # Read Example Data
    df = genfromtxt('example_data_compressed_air.csv', delimiter=',')

    P = np.array([df[1], df[2], df[3]])
    Y = np.array([df[4], df[5]])
    Ptest = np.array([df[6], df[7], df[8]])
    Ytest = np.array([df[9], df[10]])

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/compair.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_friction.py
# This is an example of a static system with one output and one input
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    P = df[1]
    Y = df[2]
    Ptest = df[3]
    Ytest = df[4]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/friction.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_narendra4.py
# This is an example of a dynamic system with one output and one delayed input
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_narendra4.csv', delimiter=',')
    P = df[1]
    Y = df[2]
    Ptest = df[3]
    Ytest = df[4]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/narendra4.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_pt2.py
# This is an example of a dynamic system with one input and one output
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    P = df[1]
    Y = df[2]
    Ptest = df[3]
    Ytest = df[4]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/pt2.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_using_P0Y0_narendra4.py
for i in range(iterations):
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

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/using_P0Y0_narendra4.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example__using_P0Y0_compair.py
# This is an example of a dynamic system with 2 outputs and 3 inputs
for i in range(iterations):
    # Read Example Data
    df = genfromtxt('example_data_compressed_air.csv', delimiter=',')

    P = np.array([df[1], df[2], df[3]])
    Y = np.array([df[4], df[5]])
    Ptest_ = np.array([df[6], df[7], df[8]])
    Ytest_ = np.array([df[9], df[10]])

    # define the first timestep t=0 of Test Data as previous (known) data P0test and Y0test
    P0test = Ptest_[:, 0:1]
    Y0test = Ytest_[:, 0:1]
    # Use the timesteps t = [1..99] as Test Data
    Ptest = Ptest_[:, 1:100]
    Ytest = Ytest_[:, 1:100]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/using_P0Y0_compair.csv")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
for i in range(iterations):

    df = genfromtxt('example_data_pt2.csv', delimiter=',')

    P = df[1]
    Y = df[2]

    # Load saved NN from file
    net = prn.loadNN("./SavedNN/gradient.csv")

    psutil.cpu_percent(interval=None, percpu=True)
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
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# preprocessing data to make it consistent (different lengths for different reviews)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


for i in range(iterations):
    model = keras.models.load_model("model.h5")

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    with open("test.txt", encoding="utf-8") as f:
        for line in f.readlines():
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace(
                "\"",
                "").strip(
                " ")
            encode = review_encode(nline)
            encode = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                                maxlen=250)
            predict = model.predict(encode)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

cores.append(psutil.cpu_percent(interval=None, percpu=True))
virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Logging data
for i in range(iterations*(len(labels)-1)):
    time_diff.append(round(time_stop[i] - time_start[i], 10))
    time_total += time_stop[i] - time_start[i]
time_diff.append(round(time_total/iterations, 10))

i = 0
for core in cores:
    cpu_percent.append(mean(cores[i]))
    i += 1
i = 0

with open('./logging/' + naam + ".csv", mode='w') as results_file:
    fieldnames = ['Naam', 'CPU Percentage', 'timediff', 'virtual mem']
    file_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    file_writer.writeheader()
    for i in range(iterations*(len(labels)-1)+1):
        j = int(i/iterations)
        file_writer.writerow({'Naam': labels[j], 'CPU Percentage':  str(cpu_percent[i]), 'timediff': str(time_diff[i]),
                              'virtual mem': str(virtual_mem[i])})


