# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim




''' Data Preprocessing '''

################ Load Data ####################################################
train = loadmat('Benchmarks/Bioreactor/APRBS_Data_3')
train = train['data']
val = loadmat('Benchmarks/Bioreactor/APRBS_Data_1')
val = val['data']
test = loadmat('Benchmarks/Bioreactor/APRBS_Data_2')
test = test['data']


################ Subsample Data ###############################################
train = train[0::50,:]
val = val[0::50,:]
test = test[0::50,:]
################# Pick Training- Validation- and Test-Data ####################

train_u = train[0:-1,0].reshape(1,-1,1)
train_y = train[1::,1].reshape(1,-1,1)

val_u = val[0:-1,0].reshape(1,-1,1)
val_y = val[1::,1].reshape(1,-1,1)

test_u = test[0:-1,0].reshape(1,-1,1)
test_y = test[1::,1].reshape(1,-1,1)

init_state = np.zeros((1,2,1))

# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'u_train':train_u[:,0:200,:], 'y_train':train_y[:,0:200,:],'init_state_train': init_state,
        'u_val':val_u[:,0:200,:], 'y_val':val_y[:,0:200,:],'init_state_val': init_state}


''' Identification '''
# Load inital linear state space model
LSS=loadmat("./Benchmarks/Bioreactor/Bioreactor_LSS")
LSS=LSS['Results']

SubspaceModel = NN.LinearSSM(dim_u=1,dim_x=2,dim_y=1)
SubspaceModel.Parameters = {'A': LSS['A'][0][0],
                  'B': LSS['B'][0][0],
                  'C': LSS['C'][0][0]}

C = LSS['C'][0][0]


x_est,y_est = SubspaceModel.Simulation(init_state[0], train_u[0])



# x_LS contains estimates for x0,...,xN
x_LS = param_optim.EstimateNonlinearStateSequence(SubspaceModel,data,10E6)

data['x_train'] = x_LS.reshape(1,-1,2)[:,0:200,:]


'''
Now data is arranged as

u0,x0,y1
...
uN,xN,yN+1

'''

# Pick a nonlinear model
model = NN.Rehmer_NN_LPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,
                         NN_A_dim=[[5,1]],NN_A_act=[[1,0]])

model.Parameters['A'] = LSS['A'][0][0]
model.Parameters['B'] = LSS['B'][0][0]
model.Parameters['C'] = LSS['C'][0][0]

model.FrozenParameters = ['A','B','C']


new_params = param_optim.ModelParameterEstimation(model,data)

for p in new_params.keys():
    model.Parameters[p] = new_params[p]



x_LS = data['x_train'][0]
u = train_u[0][0:200]
x_est = []
y_est = []


# initial states
x_est.append(init_state[0])
              
# Simulate Model
for k in range(u.shape[0]):
    x_new,y_new = model.OneStepPrediction(x_LS[k],u[[k],:])
    x_est.append(x_new)
    y_est.append(y_new)

# Concatenate list to casadiMX
y_est = cs.hcat(y_est).T    
x_est = cs.hcat(x_est).T


x_est = np.array(x_est)
y_est = np.array(y_est)

plt.close('all')

plt.figure()
plt.plot(train_y[0,0:200,:])
plt.plot(y_est[0:200,:])

x_sim,y_sim = model.Simulation(init_state[0], train_u[0])
