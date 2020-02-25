import numpy as np
from numpy import genfromtxt
import pyrenn as prn
import csv
import time
import psutil

cpu_percent = []
virtual_mem = []
time_start = []
time_stop = []
time_diff = []
time_total_start = []
time_total_end = []
iterations = 2
labels = ["compair", "friction", "narendra4", "pt2",
          "P0Y0_narendra4", "P0Y0_compair", "gradient", "Totaal"]

###
# Creating a filename

seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "MP_NN_ALL_RUN_PC"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time_total_start = time.time()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_compair.py
for i in range(iterations):
    time_start.append(time.time())

    # Read Example Data
    df = genfromtxt('example_data_compressed_air.csv', delimiter=',')

    P = np.array([df[1], df[2], df[3]])
    Y = np.array([df[4], df[5]])
    Ptest = np.array([df[6], df[7], df[8]])
    Ytest = np.array([df[9], df[10]])

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/compair.csv")

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

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

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/friction.csv")

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

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

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/narendra4.csv")

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

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

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/pt2.csv")

    # Calculate outputs of the trained NN for train and test data
    y = prn.NNOut(P, net)
    ytest = prn.NNOut(Ptest, net)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

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

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/using_P0Y0_narendra4.csv")

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

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

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/using_P0Y0_compair.csv")

    # Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
    ytest = prn.NNOut(Ptest, net)
    y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py
for i in range(iterations):
    time_start.append(time.time())

    df = genfromtxt('example_data_pt2.csv', delimiter=',')

    P = df[1]
    Y = df[2]

    # Load saved NN from file
    net = prn.loadNN("D:/School/Masterproef/Python/pyrenn/SavedNN/gradient.csv")

    # Prepare input Data for gradient calculation
    data, net = prn.prepare_data(P, Y, net)

    # Real Time Recurrent Learning
    t0_rtrl = time.time()
    J, E, e = prn.RTRL(net, data)
    g_rtrl = 2 * np.dot(J.transpose(), e)  # calculate g from Jacobian and error vector
    t1_rtrl = time.time()

    # Back Propagation Through Time
    t0_bptt = time.time()
    g_bptt, E = prn.BPTT(net, data)
    t1_bptt = time.time()

    # Compare
    # print('\n\n\nComparing Methods:')
    # print('Time RTRL: ', (t1_rtrl - t0_rtrl), 's')
    # print('Time BPTT: ', (t1_bptt - t0_bptt), 's')
    # if not np.any(np.abs(g_rtrl - g_bptt) > 1e-9):
    #    print('\nBoth methods showing the same result!')
    #    print('g_rtrl/g_bptt = ', g_rtrl / g_bptt)

    time_stop.append(time.time())
    cpu_percent.append(psutil.cpu_percent())
    virtual_mem.append(psutil.virtual_memory())

time_total_end = time.time()
cpu_percent.append(psutil.cpu_percent())
virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Logging data
for i in range(iterations*7):
    time_diff.append(time_stop[i] - time_start[i])
time_diff.append(time_total_end - time_total_start)

with open('D:/School/Masterproef/Python/pyrenn/Logging/' + naam + ".csv", mode='w') as results_file:
    fieldnames = ['Naam', 'CPU Percentage', 'timediff', 'virtual mem']
    file_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    file_writer.writeheader()
    for i in range(iterations*8-1):
        j = int(i/iterations)
        file_writer.writerow({'Naam': labels[j], 'CPU Percentage':  str(cpu_percent[i]), 'timediff': str(time_diff[i]),
                              'virtual mem': str(virtual_mem[i])})