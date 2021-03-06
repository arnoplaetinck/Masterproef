import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pyrenn as prn

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
# Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P, net)
ytest = prn.NNOut(Ptest, net)




###
# Plot results
fig = plt.figure(figsize=(15, 10))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222, sharey=ax0)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224, sharey=ax2)
fs = 18

t = np.arange(0, 11.0)/4  # 11??? (480 vorige timesteps in 15 Minute resolution
# Train Data
ax0.set_title('Train Data', fontsize=fs)


