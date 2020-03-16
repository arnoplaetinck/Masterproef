from __future__ import absolute_import, division, print_function, unicode_literals
import time
import csv
import psutil
import pyrenn as prn
from statistics import mean
from numpy import genfromtxt
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
iterations = 2
labels = ["compair", "friction", "narendra4", "pt2",
          "P0Y0_narendra4", "P0Y0_compair", "gradient", "titanic", "Totaal"]

###
# Creating a filename
seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "MP_NN_ALL_TRAIN_PC"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


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

    psutil.cpu_percent(interval=None, percpu=True)
    time_start.append(time.time())

    # Create and train NN
    net = prn.CreateNN([3, 5, 5, 2], dIn=[0], dIntern=[], dOut=[1])
    net = prn.train_LM(P, Y, net, verbose=True, k_max=500, E_stop=1e-5)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/compair.csv")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_friction.py
for i in range(iterations):
    time_start.append(time.time())

    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    P = df[1]
    Y = df[2]
    Ptest = df[3]
    Ytest = df[4]

    psutil.cpu_percent(interval=None, percpu=True)

    # Create and train NN
    net = prn.CreateNN([1, 3, 3, 1])
    net = prn.train_LM(P, Y, net, verbose=True, k_max=100, E_stop=9e-4)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/friction.csv")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_narendra4.py
for i in range(iterations):
    time_start.append(time.time())

    # Read Example Data
    df = genfromtxt('example_data_narendra4.csv', delimiter=',')
    P = df[1]
    Y = df[2]
    Ptest = df[3]
    Ytest = df[4]

    psutil.cpu_percent(interval=None, percpu=True)

    # Create and train NN
    net = prn.CreateNN([1, 3, 3, 1], dIn=[1, 2], dIntern=[], dOut=[1, 2, 3])
    net = prn.train_LM(P, Y, net, verbose=True, k_max=200, E_stop=1e-3)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())
###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/narendra4.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_pt2.py
for i in range(iterations):
    time_start.append(time.time())

    # Read Example Data
    df = genfromtxt('example_data_friction.csv', delimiter=',')
    P = df[1]
    Y = df[2]
    Ptest = df[3]
    Ytest = df[4]

    psutil.cpu_percent(interval=None, percpu=True)

    # Create and train NN
    net = prn.CreateNN([1, 2, 2, 1], dIn=[0], dIntern=[1], dOut=[1, 2])
    net = prn.train_LM(P, Y, net, verbose=True, k_max=100, E_stop=1e-3)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/pt2.csv")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_using_P0Y0_narendra4.py
for i in range(iterations):
    time_start.append(time.time())

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

    psutil.cpu_percent(interval=None, percpu=True)

    # Create and train NN
    net = prn.CreateNN([1, 3, 3, 1], dIn=[1, 2], dIntern=[], dOut=[1, 2, 3])
    net = prn.train_LM(P, Y, net, verbose=True, k_max=200, E_stop=1e-3)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/using_P0Y0_narendra4.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example__using_P0Y0_compair.py
for i in range(iterations):
    time_start.append(time.time())

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

    psutil.cpu_percent(interval=None, percpu=True)

    # Create and train NN
    net = prn.CreateNN([3, 5, 5, 2], dIn=[0], dIntern=[], dOut=[1])
    prn.train_LM(P, Y, net, verbose=True, k_max=500, E_stop=1e-5)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())

###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/using_P0Y0_compair.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
for i in range(iterations):
    time_start.append(time.time())
    df = genfromtxt('example_data_pt2.csv', delimiter=',')
    P = df[1]
    Y = df[2]

    psutil.cpu_percent(interval=None, percpu=True)

    # Create and train NN
    net = prn.CreateNN([1, 2, 2, 1], dIn=[0], dIntern=[1], dOut=[1, 2])

    # Prepare input Data for gradient calculation
    data, net = prn.prepare_data(P, Y, net)

    # Calculate derivative vector (gradient vector)
    # Real Time Recurrent Learning
    t0_rtrl = time.time()
    J, E, e = prn.RTRL(net, data)
    g_rtrl = 2 * np.dot(J.transpose(), e)  # calculate g from Jacobian and error vector
    t1_rtrl = time.time()
    # Back Propagation Through Time
    t0_bptt = time.time()
    g_bptt, E = prn.BPTT(net, data)
    t1_bptt = time.time()

    ###
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

###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/gradient.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example__using_P0Y0_compair.py
for i in range(iterations):
    time_start.append(time.time())

    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

    train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
    test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)

    ###
    # Load data
    LABEL_COLUMN = 'survived'
    LABELS = [0, 1]

    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
    DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
    temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS, column_defaults=DEFAULTS)

    example_batch, labels_batch = next(iter(temp_dataset))
    packed_dataset = temp_dataset.map(pack)

    NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']

    packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
    packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

    example_batch, labels_batch = next(iter(packed_train_data))

    # normalizing continuous data
    desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])

    # See what you just created.
    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer,
                                                      shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]
    numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
    numeric_layer(example_batch).numpy()

    CATEGORIES = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
    }

    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    # See what you just created.
    categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

    # Building the model
    model = tf.keras.Sequential(
        [preprocessing_layer, tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dense(1), ])
    new_model = tf.keras.Sequential(
        [preprocessing_layer, tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dense(1), ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    train_data = packed_train_data.shuffle(500)
    test_data = packed_test_data

    psutil.cpu_percent(interval=None, percpu=True)

    model.fit(train_data, epochs=5)

    time_stop.append(time.time())
    cores.append(psutil.cpu_percent(interval=None, percpu=True))
    virtual_mem.append(psutil.virtual_memory())
###
# Save outputs to certain file
model.save_weights('./SavedNN/titanic/saved_weights')

cores.append(psutil.cpu_percent(interval=None, percpu=True))
virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Logging data
for i in range(iterations*(len(labels)-1)):
    time_diff.append(time_stop[i] - time_start[i])
    time_total += time_stop[i] - time_start[i]
time_diff.append(time_total/iterations)
i = 0
for core in cores:
    cpu_percent.append(mean(cores[i]))
    i += 1
i = 0

with open('D:/School/Masterproef/Python/pyrenn/Logging/' + naam + ".csv", mode='w') as results_file:
    fieldnames = ['Naam', 'CPU Percentage', 'timediff', 'virtual mem']
    file_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    file_writer.writeheader()
    for i in range(iterations*(len(labels)-1)+1):
        j = int(i/iterations)
        file_writer.writerow({'Naam': labels[j], 'CPU Percentage':  str(cpu_percent[i]), 'timediff': str(time_diff[i]),
                              'virtual mem': str(virtual_mem[i])})
print(cores)