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
data = {'u_train':train_u, 'y_train':train_y,'init_state_train': init_state,
        'u_val':val_u, 'y_val':val_y,'init_state_val': init_state}


''' Identification '''
# Load inital linear state space model
LSS=loadmat("./Benchmarks/Bioreactor/Bioreactor_LSS")
LSS=LSS['Results']


''' RBF approach'''

initial_params = {'A': LSS['A'][0][0],
                  'B': LSS['B'][0][0],
                  'C': LSS['C'][0][0],
                  'range_u': np.array([[0,0.7]]),
                  'range_x': np.array([[-0.1,0.1],[-0.1,0.1]])}

p_opts = {"expand":False}
# s_opts = {"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}


''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''
model = NN.Rehmer_NN_LPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,dim_thetaB=1,
                          dim_thetaC=2, NN_A_dim=[[5,1],[2,1]],NN_B_dim=[[3,1]],
                          NN_C_dim=[[3,1],[1,1]], NN_A_act=[[0,0],[0,0]],
                          NN_B_act=[[0,0]], NN_C_act=[[0,0],[0,0]],
                          initial_params=None,init_proc='random')
# 
x,y = model.Simulation(np.array([[0],[0]]),train_u[0])





