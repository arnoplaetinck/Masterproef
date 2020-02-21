import numpy as np
from numpy import genfromtxt
import pyrenn as prn
import logging
import time
import psutil

cpu_percent = [0, 0, 0, 0, 0, 0, 0, 0]
virtual_mem = [0, 0, 0, 0, 0, 0, 0, 0]
time_start = [0, 0, 0, 0, 0, 0, 0, 0]
time_stop = [0, 0, 0, 0, 0, 0, 0, 0]
time_diff = [0, 0, 0, 0, 0, 0, 0, 0]
labels = ["example_compair.py", "example_friction.py", "example_narendra4.py", "example_pt2.py",
          "example_using_P0Y0_narendra4.py", "example_using_P0Y0_compair.py", "example_gradient.py"]
cpu_percent[7] = psutil.cpu_percent()
virtual_mem[7] = psutil.virtual_memory()

seconds = time.time()
local_time = time.ctime(seconds)
naam2 = local_time.split()
naam = "MP_NN_ALL"
for i in range(len(naam2)):
    naam += "_" + naam2[i]
naam = naam.replace(':', '_')

time_start[7] = time.time()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_compair.py

time_start[0] = time.time()

###
# Read Example Data
df = genfromtxt('example_data_compressed_air.csv', delimiter=',')

P = np.array([df[1], df[2], df[3]])
Y = np.array([df[4], df[5]])
Ptest = np.array([df[6], df[7], df[8]])
Ytest = np.array([df[9], df[10]])

###
# Create and train NN
# create feed forward neural network with 1 input, 2 hidden layers with
# 4 neurons each and 1 output
# the NN has a recurrent connection with delay of 1 timesteps from the output
# to the first layer
net = prn.CreateNN([3, 5, 5, 2], dIn=[0], dIntern=[], dOut=[1])
# Train NN with training data P=input and Y=target
# Set maximum number of iterations k_max to 500
# Set termination condition for Error E_stop to 1e-5
# The Training will stop after 500 iterations or when the Error <=E_stop
net = prn.train_LM(P, Y, net, verbose=True, k_max=500, E_stop=1e-5)

###
# Save outputs to certain file
prn.saveNN(net, "D:/School/Masterproef/Python/pyrenn/SavedNN/compair.csv")
print("savegelukt")

###
# Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P, net)
ytest = prn.NNOut(Ptest, net)

time_stop[0] = time.time()
cpu_percent[0] = psutil.cpu_percent()
virtual_mem[0] = psutil.virtual_memory()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_friction.py

time_start[1] = time.time()
###
# Read Example Data
df = genfromtxt('example_data_friction.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]
###
# Create and train NN

# create feed forward neural network with 1 input, 2 hidden layers with
# 3 neurons each and 1 output
net = prn.CreateNN([1, 3, 3, 1])

# Train NN with training data P=input and Y=target
# Set maximum number of iterations k_max to 100
# Set termination condition for Error E_stop to 1e-5
# The Training will stop after 100 iterations or when the Error <=E_stop
net = prn.train_LM(P, Y, net, verbose=True, k_max=100, E_stop=9e-4)

###
# Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P, net)
ytest = prn.NNOut(Ptest, net)

time_stop[1] = time.time()
cpu_percent[1] = psutil.cpu_percent()
virtual_mem[1] = psutil.virtual_memory()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_narendra4.py

time_start[2] = time.time()
###
# Read Example Data
df = genfromtxt('example_data_narendra4.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]
###
# Create and train NN
# create recurrent neural network with 1 input, 2 hidden layers with
# 3 neurons each and 1 output
# the NN uses the input data at timestep t-1 and t-2
# The NN has a recurrent connection with delay of 1,2 and 3 timesteps from the output
# to the first layer (and no recurrent connection of the hidden layers)
net = prn.CreateNN([1, 3, 3, 1], dIn=[1, 2], dIntern=[], dOut=[1, 2, 3])

# Train NN with training data P=input and Y=target
# Set maximum number of iterations k_max to 200
# Set termination condition for Error E_stop to 1e-3
# The Training will stop after 200 iterations or when the Error <=E_stop
prn.train_LM(P, Y, net, verbose=True, k_max=200, E_stop=1e-3)

###
# Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P, net)
ytest = prn.NNOut(Ptest, net)

time_stop[2] = time.time()
cpu_percent[2] = psutil.cpu_percent()
virtual_mem[2] = psutil.virtual_memory()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_pt2.py

time_start[3] = time.time()
###
# Read Example Data
df = genfromtxt('example_data_friction.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]

###
# Create and train NN

# create recurrent neural network with 1 input, 2 hidden layers with
# 2 neurons each and 1 output
# the NN has a recurrent connection with delay of 1 timestep in the hidden
# layers and a recurrent connection with delay of 1 and 2 timesteps from the output
# to the first layer
net = prn.CreateNN([1, 2, 2, 1], dIn=[0], dIntern=[1], dOut=[1, 2])

# Train NN with training data P=input and Y=target
# Set maximum number of iterations k_max to 100
# Set termination condition for Error E_stop to 1e-3
# The Training will stop after 100 iterations or when the Error <=E_stop
net = prn.train_LM(P, Y, net, verbose=True, k_max=100, E_stop=1e-3)

###
# Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P, net)
ytest = prn.NNOut(Ptest, net)

time_stop[3] = time.time()
cpu_percent[3] = psutil.cpu_percent()
virtual_mem[3] = psutil.virtual_memory()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_using_P0Y0_narendra4.py

time_start[4] = time.time()
###
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

###
# Create and train NN

# create recurrent neural network with 1 input, 2 hidden layers with
# 3 neurons each and 1 output
# the NN uses the input data at timestep t-1 and t-2
# The NN has a recurrent connection with delay of 1,2 and 3 timesteps from the output
# to the first layer (and no recurrent connection of the hidden layers)
net = prn.CreateNN([1, 3, 3, 1], dIn=[1, 2], dIntern=[], dOut=[1, 2, 3])

# Train NN with training data P=input and Y=target
# Set maximum number of iterations k_max to 200
# Set termination condition for Error E_stop to 1e-3
# The Training will stop after 200 iterations or when the Error <=E_stop
net = prn.train_LM(P, Y, net, verbose=True, k_max=200, E_stop=1e-3)

###
# Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
ytest = prn.NNOut(Ptest, net)
y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

time_stop[4] = time.time()
cpu_percent[4] = psutil.cpu_percent()
virtual_mem[4] = psutil.virtual_memory()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example__using_P0Y0_compair.py

time_start[5] = time.time()

###
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

###
# Create and train NN

# create feed forward neural network with 1 input, 2 hidden layers with
# 4 neurons each and 1 output
# the NN has a recurrent connection with delay of 1 timesteps from the output
# to the first layer
net = prn.CreateNN([3, 5, 5, 2], dIn=[0], dIntern=[], dOut=[1])

# Train NN with training data P=input and Y=target
# Set maximum number of iterations k_max to 500
# Set termination condition for Error E_stop to 1e-5
# The Training will stop after 500 iterations or when the Error <=E_stop
prn.train_LM(P, Y, net, verbose=True, k_max=500, E_stop=1e-5)

###
# Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
ytest = prn.NNOut(Ptest, net)
y0test = prn.NNOut(Ptest, net, P0=P0test, Y0=Y0test)

time_stop[5] = time.time()
cpu_percent[5] = psutil.cpu_percent()
virtual_mem[5] = psutil.virtual_memory()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# example_gradient.py

time_start[6] = time.time()

df = genfromtxt('example_data_pt2.csv', delimiter=',')

P = df[1]
Y = df[2]

###
# Create and train NN

# create recurrent neural network with 1 input, 2 hidden layers with
# 2 neurons each and 1 output
# the NN has a recurrent connection with delay of 1 timestep in the hidden
# layers and a recurrent connection with delay of 1 and 2 timesteps from the output
# to the first layer
net = prn.CreateNN([1, 2, 2, 1], dIn=[0], dIntern=[1], dOut=[1, 2])

###
# Prepare input Data for gradient calculation
data, net = prn.prepare_data(P, Y, net)

###
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

time_stop[6] = time.time()
cpu_percent[6] = psutil.cpu_percent()
virtual_mem[6] = psutil.virtual_memory()

time_stop[7] = time.time()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Logging data
for i in range(8):
    time_diff[i] = time_stop[i] - time_start[i]

f = open('D:/School/Masterproef/Python/pyrenn/Logging/' + naam + ".txt", "a+")
for i in range(7):
    f.write(labels[i]+": " + str(cpu_percent[i]) + " time: " + str(time_diff[i]) + "  " + str(virtual_mem[i]) + "\n")
f.write("Total time: " + str(time_diff[7])+ " cpu aan begin :"+ str(cpu_percent[7]) +" vm aan begin :" + str(virtual_mem[7]))
f.close()
