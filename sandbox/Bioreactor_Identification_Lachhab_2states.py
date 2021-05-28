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

initial_params = {'A_0': LSS['A'][0][0],
                  'B_0': LSS['B'][0][0],
                  'C_0': LSS['C'][0][0]}


''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''

for dim in [1,2,3,4,5]:
    
    model = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,dim_thetaB=dim,
                      dim_thetaC=0,initial_params=initial_params,
                      name='RBF_network') 
    

    
    identification_results = param_optim.ModelTraining(model,data,10,
                             initial_params=initial_params,p_opts=None,
                             s_opts=None)

    pkl.dump(identification_results,open('Bioreactor_Lachhab_2states_theta'+str(dim)+'.pkl',
                                         'wb'))
