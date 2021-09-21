# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:23:32 2021

@author: alexa
"""

# Estimate linear model (=import from matlab)

'''Estimate nonlinear state sequence'''
# Take linear model (Linear SS)
# Simulate it
# Estimate nonlinear state sequence recursively
# vary trade-off parameter lambda

''' ''' 

# Train Autoencoder (nonlinear static regression) and parameter map

# Optimize whole model

# 

# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim


''' Generate measurement data for first phase '''
N = 100

np.random.seed(42)

u = np.random.normal(0,1,(1,N-1,1))
x = np.zeros((1,N,1))
y = np.zeros((1,N,1))

y[0,0,0] = x[0,0,0]**3

for k in range(1,N):
    x[0,k,0] = 0.7*x[0,k-1,0] + 1*u[0,k-1,0] 
    y[0,k,0] = x[0,k,0]**3 
    



init_state = x[0,0,0].reshape(1,1,1) 
data = {'u_train':u, 'y_train':y,'init_state_train': init_state}


model = NN.LinearSSM(dim_u=1,dim_x=1,dim_y=1,name='test')

model.Parameters = {'A': np.array([[0.7]]),
                    'B': np.array([[1]]),
                    'C': np.array([[6]])}

x_est,y_est = model.Simulation(init_state[0], u[0])


# plt.plot(np.array(x[0]))
# plt.plot(np.array(x_est))


# plt.plot(np.array(y[0,1::,:]))
# plt.plot(np.array(y_est))


x_LS_0 = param_optim.EstimateNonlinearStateSequence(model,data,0)

x_LS_1000 = param_optim.EstimateNonlinearStateSequence(model,data,100000)

plt.figure()
plt.plot(np.array(x[0]))
plt.plot(x_LS_1000)

plt.figure()
plt.plot(x_LS_0*6)
plt.plot(data['y_train'][0])


