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
train = loadmat('Benchmarks/RobotManipulator/APRBS_Ident_Data')
train = train['data']
val = loadmat('Benchmarks/RobotManipulator/APRBS_Val_Data')
val = val['data']


################# Pick Training- Validation- and Test-Data ####################

train_u = train[0:5000,0:2].reshape(1,-1,2)
train_y = train[0:5000,2::].reshape(1,-1,2)

val_u = train[0:1000,0:2].reshape(1,-1,2)
val_y = val[0:1000,2::].reshape(1,-1,2)


init_state = np.zeros((1,4,1))


# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'u_train':train_u, 'y_train':train_y,'init_state_train': init_state,
        'u_val':val_u, 'y_val':val_y,'init_state_val': init_state}


''' Identification '''
# Load inital linear state space model
LSS=loadmat("./Benchmarks/RobotManipulator/RobotManipulator_LSS")
LSS=LSS['Results']


''' Approach Rehmer '''
initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
model = NN.RehmerLPV(dim_u=2,dim_x=4,dim_y=2,dim_thetaA=2,dim_thetaB=0,
                          dim_thetaC=0,fA_dim=20,fB_dim=0,fC_dim=0,
                          initial_params=initial_params,name='Rehmer_LPV')

''' Approach Lachhab '''
# initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
# model = Model.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=2,dim_thetaB=0,
#                           dim_thetaC=0,name='Lachhab_LPV')


''' RBF approach'''
# initial_params = {'A0': LSS['A'][0][0], 'B0': LSS['B'][0][0], 'C0': LSS['C'][0][0],
#                   'A1': LSS['A'][0][0], 'B1': LSS['B'][0][0], 'C1': LSS['C'][0][0]}
# model = Model.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=2,
#                       initial_params=initial_params,name='RBF_network')

''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''
identification_results = param_optim.ModelTraining(model,data,1,initial_params=initial_params)

pkl.dump(identification_results,open('RobotMan_theta2_Neur20_tanh.pkl','wb'))
# identification_results = pkl.load(open('Benchmarks/Silverbox/IdentifiedModels/Silverbox_Topmodel.pkl','rb'))

''' The output is a pandas dataframe which contains the results for each of
the 10 initializations, specifically the loss on the validation data
and the estimated parameters ''' 

# Pick the parameters from the first initialization (for example, in this case
# every model has a loss close to zero because the optimizer is really good
# and its 'only' a linear model which we identify)

# model.Parameters = identification_results.loc[0,'params']


# Maybe plot the simulation result on test data to see how good the model performs
# x_est,y_est = model.Simulation(init_state[0],train_u[0,0:1000])

# y_est = np.array(y_est) 
 
 
# plt.plot(train_y[0,0:5000],label='True output')                                        # Plot True data
# plt.plot(y_est,label='Est. output')                                            # Plot Model Output
# plt.plot(val_y[0]-y_est,label='Simulation Error')                             # Plot Error between model and true system (its almost zero)
# plt.legend()
# plt.show()


# Scatterplot of affine Parameters for visual inspection

# theta = np.array(theta) 
# plt.figure()
# plt.plot(thetaA[:,0],label='Theta_A1')    
# plt.plot(thetaA[:,1],label='Theta_A2')   
# plt.scatter(theta[:,0],theta[:,1])  
# plt.legend()
# plt.show()
# e2 = y[0]-y_est

# model.AffineStateSpaceMatrices([1,1])
