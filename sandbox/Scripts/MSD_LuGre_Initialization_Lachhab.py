# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim

''' User specified parameters '''
dim_x = 3
inits = 1
lamb = 0.01

''' Data Preprocessing '''

################ Load Data ####################################################
train = loadmat('Benchmarks/Mass_Spring_Damper_LuGre/data/dataset1')
train = train['dataset1']
val = loadmat('Benchmarks/Mass_Spring_Damper_LuGre/data/dataset2')
val = val['dataset2']
test = loadmat('Benchmarks/Mass_Spring_Damper_LuGre/data/dataset3')
test = test['dataset3']


################# Pick Training- Validation- and Test-Data ####################

train_u = train[0:-1,0].reshape(1,-1,1)     # shift data
train_y = train[1::,1].reshape(1,-1,1)      # shift data

val_u = val[0:-1,0].reshape(1,-1,1)
val_y = val[1::,1].reshape(1,-1,1)

test_u = test[0:-1,0].reshape(1,-1,1)
test_y = test[1::,1].reshape(1,-1,1)

init_state = np.zeros((1,dim_x,1)) # system was excited from zero position

# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'u_train':train_u[0:2000], 'y_train':train_y[0:2000],'init_state_train': init_state,
        'u_val':val_u[0:2000] , 'y_val':val_y[0:2000],'init_state_val': init_state,
        'u_test':test_u[0:2000] , 'y_test':test_y[0:2000],'init_state_test': init_state}


''' Pre-Identification via estimated state sequence '''
# Load inital linear state space model
LSS=loadmat("Benchmarks/Mass_Spring_Damper_LuGre/data/LuGre_LSS_s3")
LSS=LSS['LuGre_LSS']

SubspaceModel = NN.LinearSSM(dim_u=1,dim_x=dim_x,dim_y=1)
SubspaceModel.Parameters = {'A': LSS[0][0][0],
                  'B': LSS[0][0][1],
                  'C': LSS[0][0][2],
                  'D': LSS[0][0][3]}

# Estimate nonlinear state sequence
x_LS = param_optim.EstimateNonlinearStateSequence(SubspaceModel,data,lamb)

# Add state sequence to data
data['x_train'] = x_LS.reshape(1,-1,dim_x)


# p_opts = {"expand":False}
s_opts = {"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}


model = NN.LachhabLPV(dim_u=1,dim_x=dim_x,dim_y=1, dim_thetaA=1)
            
model.FrozenParameters = ['A_0','B_0','C_0']
    
model.InitialParameters = initial_params = {'A_0': LSS['A'][0][0],
                                            'B_0': LSS['B'][0][0],
                                            'C_0': LSS['C'][0][0]}
    
results = param_optim.ModelTraining(model,data,inits,
                         p_opts=None,s_opts=s_opts,mode='series')
                


