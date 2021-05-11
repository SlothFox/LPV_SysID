# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:32:58 2021

@author: alexa
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat, savemat

from models.RealWorldSystems import RobotManipulator,RobotManipulator2,Bioreactor
from testsignals.testsignals import APRBS



''' Design Input Signal '''
N = 600
# step_range = [0,0.6]
# holding_range = [25*50,50*50]
# u = APRBS(N,step_range,holding_range)

# u = u.reshape(N,1)

u = np.ones((N,1))*1.29133


''' Initialize Plant'''
model = Bioreactor('Bioreactor')

''' Simulate Plant '''

x = np.zeros((N,2))
y = np.zeros((N,1))


x[0] = [0.22,0.69894]

G = 0.48
b = 0.02
T = 0.01

for k in range(0,N-1):

    x_new,y_new = model.OneStepPrediction(x[k],u[k])

    x[k+1] = np.array(x_new).T
    y[k+1] = np.array(y_new).T


# u = np.vstack(u)     
# x = np.vstack(x)  
# y = np.vstack(y)     
    
plt.close('all')

plt.figure()
# plt.plot(u,label='u')
plt.plot(x,label='x')
# plt.plot(y,label='y')
# plt.legend()


# plt.figure()
# plt.scatter(y[:,0],y[:,1])
# plt.xlim([-2,2])
# plt.ylim([-2,2])


# io_data = np.hstack((u[:-1],y))

# mdic = {"data": io_data, "label": "APRBS_Val_Data"}

# RobotManipulator2

# savemat('Benchmarks/RobotManipulator/APRBS_Val_Data.mat',mdic)
