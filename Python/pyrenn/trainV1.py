import time
import csv
import numpy as np
import psutil
import pyrenn as prn
from statistics import mean
from numpy import genfromtxt

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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# first time calling cpu percent to get rid of 0,0
psutil.cpu_percent(interval=None, percpu=True)
time_total_start = time.time()

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
prn.saveNN(net, "./SavedNN/compair.csv")


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
prn.saveNN(net, "./SavedNN/friction.csv")


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
prn.saveNN(net, "./SavedNN/narendra4.csv")

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
prn.saveNN(net, "./SavedNN/pt2.csv")
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
prn.saveNN(net, "./SavedNN/using_P0Y0_narendra4.csv")

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
prn.saveNN(net, "./SavedNN/using_P0Y0_compair.csv")

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
prn.saveNN(net, "./SavedNN/gradient.csv")

time_total_end = time.time()
cores.append(psutil.cpu_percent(interval=None, percpu=True))
virtual_mem.append(psutil.virtual_memory())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Logging data
for i in range(iterations*7):
    time_diff.append(time_stop[i] - time_start[i])
    time_total += time_stop[i] - time_start[i]
time_diff.append((time_total_end - time_total_start)/iterations)
i = 0
for core in cores:
    cpu_percent.append(mean(cores[i]))
    i += 1
i = 0

with open('./logging/' + naam + ".csv", mode='w') as results_file:
    fieldnames = ['Naam', 'CPU Percentage', 'timediff', 'virtual mem']
    file_writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    file_writer.writeheader()
    for i in range(iterations*7+1):
        j = int(i/iterations)
        file_writer.writerow({'Naam': labels[j], 'CPU Percentage':  str(cpu_percent[i]), 'timediff': str(time_diff[i]),
                              'virtual mem': str(virtual_mem[i])})
print(cores)