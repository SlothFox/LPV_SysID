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

from scipy.signal import decimate
from scipy.fft import fft

from models.RealWorldSystems import RobotManipulator,RobotManipulator2,Bioreactor
from testsignals.testsignals import APRBS



''' Design Input Signal '''
N = 50000
step_range = [0.0,0.7]
holding_range = [25*50,50*50]
u = APRBS(N,step_range,holding_range)

u = u.reshape(N,1)

# u = np.zeros((N,1))*1.29133


''' Initialize Plant'''
model = Bioreactor('Bioreactor')

''' Simulate Plant '''

x = np.zeros((N,2))
y = np.zeros((N,1))


# x[0] = [0.22,0.69894]
x[0] = [0.107,1]

# G = 0.48
# b = 0.02
# T = 0.01

for k in range(0,N-1):
    # print(x)
    x_new,y_new = model.OneStepPrediction(x[k],u[k])
    
    # term = x[k-1,0]*(1-x[k-1,1])*np.exp(x[k-1,1]/G)
    
    # x[k,0] = x[k-1,0] + T * (-x[k-1,0]*u[k-1] + term)
    # x[k,1] = x[k-1,1] + T * (-x[k-1,1]*u[k-1] + term) * ((1+b)/(1+b-x[k-1,1]))
    
    x[k+1] = np.array(x_new).T
    # print(x)
    y[k+1] = np.array(y_new).T


# u = np.vstack(u)     
# x = np.vstack(x)  
# y = np.vstack(y)     
    
# plt.close('all')

plt.figure()
plt.plot(u,label='u')
plt.plot(x,label='x')
# plt.plot(y,label='y')
# plt.legend()


# plt.figure()
# plt.scatter(y[:,0],y[:,1])
# plt.xlim([-2,2])
# plt.ylim([-2,2])

# y_filt= y[:,0]
# y_filt = decimate(y_filt,10)
# y_filt = decimate(y_filt,5)


# Y = np.abs(fft(y))

# time_step = 0.01
# freqs = np.fft.fftfreq(N, time_step)
# idx = np.argsort(freqs)

# plt.figure()
# plt.plot(freqs[idx], Y[idx])
# plt.plot(Y)

# y = y[0::50]
# u = u[0::50]


io_data = np.hstack((u,y))

mdic = {"data": io_data, "label": "APRBS_Data"}

savemat('Benchmarks/Bioreactor/APRBS_Data_3.mat',mdic)



