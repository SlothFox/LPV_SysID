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
inits = 2

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
data = {'u_train':train_u, 'y_train':train_y,'init_state_train': init_state,
        'u_val':val_u , 'y_val':val_y,'init_state_val': init_state,
        'u_test':test_u , 'y_test':test_y,'init_state_test': init_state}


''' Pre-Identification via estimated state sequence '''

# Load inital linear state space model
LSS=loadmat("Benchmarks/Mass_Spring_Damper_LuGre/data/LuGre_LSS_s3")
LSS=LSS['LuGre_LSS']

SubspaceModel = NN.LinearSSM(dim_u=1,dim_x=dim_x,dim_y=1)
SubspaceModel.Parameters = {'A': LSS[0][0][0],
                  'B': LSS[0][0][1],
                  'C': LSS[0][0][2]}

# Estimate nonlinear state sequence
x_LS = param_optim.EstimateNonlinearStateSequence(SubspaceModel,data,0.1)

# Add state sequence to data
data['x_train'] = x_LS.reshape(1,-1,dim_x)

initial_params = {'A': LSS['A'][0][0],
                  'B': LSS['B'][0][0],
                  'C': LSS['C'][0][0],
                  'A_0':np.array([[0,0,0],[0,0,0],[0,1,0]])}

# p_opts = {"expand":False}
s_opts = {"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}


model_structures = [1]

# dim_thetaA = [1,2,3]
# A_0s = [np.identity(9)[i,:].reshape((dim_x,dim_x)) for i in range(0,dim_x**2)]
# activations = [ [[1,0,1,0]], [[3,0,3,0]], [[1,2,1,2]]      ]
# dim_phis = [1,2,3,4]

dim_thetaA = [1]
A_0s = [np.identity(9)[i,:].reshape((dim_x,dim_x)) for i in range(0,2)]
activations = [ [[1,0,1,0]]]
dim_phis = [1,2]

for dimA in dim_thetaA:
    for A_0 in A_0s:
        for activation in activations:
            for dim_phi in dim_phis:
    
                model = NN.Rehmer_NN_LPV(dim_u=1,dim_x=dim_x,dim_y=1,
                                         dim_thetaA=dimA, NN_A_dim=[[5,dim_phi,5,1]],
                                         NN_A_act=activation)
                
                model.FrozenParameters = ['A','B','C','A_0']
                model.InitialParameters = initial_params
                
                identification_results = param_optim.ModelTraining(model,data,inits,
                                         p_opts=None,s_opts=s_opts)
                
                # Add information
                identification_results['dim_phi'] = dim_phi
                identification_results['activations'] = [activation for i in range(0,inits)]
                identification_results['dim_thetaA'] = dimA
                
                # TO DO: ALLE DATAFRAMES ANEINANDERHÄNGEN

    # pkl.dump(identification_results,open('Home_Bioreactor_RBF_2states_theta'+str(dim)+'.pkl',
    #                                      'wb'))
