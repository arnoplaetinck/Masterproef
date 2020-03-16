import numpy as np
from numpy import genfromtxt
import pyrenn as prn
import matplotlib.pyplot as plt

###
#Read Example Data
df = genfromtxt('example_data_friction.csv', delimiter=',')
P = df[1]
Y = df[2]
Ptest = df[3]
Ytest = df[4]


###
#Create and train NN

#create recurrent neural network with 1 input, 2 hidden layers with 
#2 neurons each and 1 output
#the NN has a recurrent connection with delay of 1 timestep in the hidden
# layers and a recurrent connection with delay of 1 and 2 timesteps from the output
# to the first layer
net = prn.CreateNN([1,2,2,1],dIn=[0],dIntern=[1],dOut=[1,2])

#Train NN with training data P=input and Y=target
#Set maximum number of iterations k_max to 100
#Set termination condition for Error E_stop to 1e-3
#The Training will stop after 100 iterations or when the Error <=E_stop
net = prn.train_LM(P,Y,net,verbose=True,k_max=100,E_stop=1e-3) 


###
#Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P,net)
ytest = prn.NNOut(Ptest,net)

###
#Plot results
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(y,color='b',lw=2,label='NN Output')
ax0.plot(Y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(ytest,color='b',lw=2,label='NN Output')
ax1.plot(Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()

fig.tight_layout()
plt.show()
