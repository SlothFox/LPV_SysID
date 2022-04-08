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

# from scipy.io import loadmat
from scipy.linalg import inv

import sys
# sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
# sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')
# sys.path.insert(0, '/home/alexander/GitHub/DigitalTwinInjectionMolding/')
sys.path.insert(0, 'E:/GitHub/LPV_SysID/sandbox/')

import models.NN as NN
from optim import param_optim
from testsignals.testsignals import APRBS

# Initialiaze NL system
nl_system = NN.DummySystem1(dim_u=1,dim_x=1,dim_y=1,u_lab=['u'],y_lab=['y'],initial_params=None, 
             frozen_params = [], init_proc='random')

nl_system.Parameters = {'A': np.array([[1]]),
                        'B': np.array([[0.5]]),
                        'C': np.array([[1]]),
                        'D': np.array([[0]])}

# Generate identifikation data
N = 1000

np.random.seed(42)

x0 = np.ones((1,1))*0
u = np.random.normal(0,1,(N,1))
# u = APRBS(1000,[-2,2],[100,200]).T
x,y = nl_system.Simulation(x0, u)

x = np.array(x)[0:-1]
y = np.vstack((x0,np.array(y)))[0:-1]

io_data = pd.DataFrame(data=np.hstack([u,x,y]),columns=['u','x','y'])

init_state = x[0,0].reshape(1,1) 




# Estimate Parameters of a linear SSM x_new = a*x+b*u

X = io_data.iloc[0:-1][['x','u']].values
Y = io_data.iloc[1::][['x']].values

theta = inv((X.T).dot(X)).dot(X.T).dot(Y)


lin_model = NN.LinearSSM(dim_u=1,dim_x=1,dim_y=1,u_lab=['u'],y_lab=['y'])

lin_model.Parameters = {'A': theta[[0]],
                        'B': theta[[1]],
                        'C': np.array([[1]]),
                        'D': np.array([[0]])}

# Estimate the state space sequence
data = {'data':[io_data],'init_state': [init_state]}

x_LS = param_optim.EstimateNonlinearStateSequence(lin_model,data,1)

io_data['x_ref']=x_LS['x_LS'].values

# Now estimate parameters of a model given the state space sequence
nl_system_est = NN.DummySystem2(dim_u=1,dim_x=1,dim_y=1,dim_h=5,u_lab=['u'],y_lab=['y'], 
             frozen_params = ['A','B','C','D'], init_proc='random')

initial_params = {'A': theta[[0]],
                  'B': theta[[1]],
                  'C': np.array([[1]]),
                  'D': np.array([[0]])}

nl_system_est.InitialParameters = initial_params

res = param_optim.ModelTraining(nl_system_est,data,data,initializations=1,p_opts=None,
                                s_opts=None,mode='series')


nl_system_est.Parameters.update(res.iloc[0]['params_val'])


# CHECK HERE HOW MODEL PERFORMS AND PLOT NONLINEARITY



# Simulate the linear model to get the linear state sequence
# x_est,y_est = lin_model.Simulation(init_state[0],u)

# x_est = np.array(x_est)[0:-1]
# y_est = np.vstack((x0,np.array(y_est)))[0:-1]

# io_data_est = pd.DataFrame(data=np.hstack([u,x_est,y_est]),
#                            columns=['u','x_est','y_est'])

# plt.close('all')
# plt.figure()
# plt.plot(io_data['x'],label='x')
# plt.plot(io_data['y'],label='y')
# plt.plot(x_LS['x_LS'],label='x_LS')
# plt.plot(io_data_est['x_est'],label='x_est')
# plt.legend()
# # x_LS_1000 = param_optim.EstimateNonlinearStateSequence(model,data,100000)

# plt.figure()
# plt.scatter(io_data['x'].iloc[0:-1], io_data['x'].iloc[1::])
# plt.scatter(x_LS['x_LS'].iloc[0:-1], x_LS['x_LS'].iloc[1::])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(io_data['x'].iloc[0:-1],io_data['u'].iloc[0:-1], io_data['x'].iloc[1::])
# ax.scatter(x_LS['x_LS'].iloc[0:-1],io_data['u'].iloc[0:-1], x_LS['x_LS'].iloc[1::])
# ax.scatter(io_data_est['x_est'].iloc[0:-1],io_data_est['u'].iloc[0:-1], io_data_est['x_est'].iloc[1::])
# plt.plot(np.array(x[0]))
# plt.plot(np.array(x_est))


# plt.plot(np.array(y[0,1::,:]))
# plt.plot(np.array(y_est))

# plt.figure()
# plt.plot(io_data['x'])
# plt.plot(x_LS_0)

# plt.figure()
# plt.plot(io_data['y'])
# plt.plot(x_LS_0['x_LS']*6)



