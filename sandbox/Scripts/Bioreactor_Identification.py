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

# train_u = train[:,0].reshape(1,-1,1)
# train_y = train[:,1].reshape(1,-1,1)

val_u = val[0:-1,0].reshape(1,-1,1)
val_y = val[1::,1].reshape(1,-1,1)

test_u = test[0:-1,0].reshape(1,-1,1)
test_y = test[1::,1].reshape(1,-1,1)

init_state = np.array([[[0.2059],[-0.0968]]])

# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'u_train':train_u, 'y_train':train_y,'init_state_train': init_state,
        'u_val':val_u, 'y_val':val_y,'init_state_val': init_state}


''' Identification '''
# Load inital linear state space model
LSS=loadmat("./Benchmarks/Bioreactor/Bioreactor_LSS")
LSS=LSS['Results']


''' Approach Rehmer '''
# initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
# model = NN.RehmerLPV(dim_u=2,dim_x=4,dim_y=2,dim_thetaA=0,dim_thetaB=0,
#                           dim_thetaC=0,fA_dim=0,fB_dim=0,fC_dim=0,
#                           initial_params=initial_params,name='Rehmer_LPV')

''' Approach Lachhab '''
# initial_params = {'A_0': LSS['A'][0][0], 'B_0': LSS['B'][0][0], 'C_0': LSS['C'][0][0] }
# model = Model.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=2,dim_thetaB=0,
#                           dim_thetaC=0,name='Lachhab_LPV')


''' RBF approach'''
model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=1,
                      initial_params=None,name='RBF_network')
model.InitializeLocalModels(LSS['A'][0][0],LSS['B'][0][0],LSS['C'][0][0],
                            range_u = np.array([[0,0.7]]),
                            range_x = np.array([[0.004,0.231],[-0.248,0.0732]]))

''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''

# Solver options
# p_opts = {"expand":False}
# s_opts = {"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}
    
# identification_results = param_optim.ModelTraining(model,data,100,
#                                                   initial_params=initial_params,
#                                                   p_opts=p_opts,s_opts=s_opts)

# pkl.dump(identification_results,open('RobotMan_theta2_Neur20_tanh.pkl','wb'))
# identification_results = pkl.load(open('Benchmarks/Silverbox/IdentifiedModels/Silverbox_Topmodel.pkl','rb'))

''' The output is a pandas dataframe which contains the results for each of
the 10 initializations, specifically the loss on the validation data
and the estimated parameters ''' 

# Pick the parameters from the first initialization (for example, in this case
# every model has a loss close to zero because the optimizer is really good
# and its 'only' a linear model which we identify)

# model.Parameters = identification_results.loc[0,'params']


# Maybe plot the simulation result on test data to see how good the model performs
# x_est,y_est = model.OneStepPrediction(init_state[0],train_u[0,0])
x_est,y_est = model.Simulation(init_state[0],train_u[0])

y_est = np.array(y_est) 
 
 
plt.plot(train_y[0],label='True output')                                       # Plot True data
plt.plot(y_est,label='Est. output')                                            # Plot Model Output
# plt.plot(val_y[0]-y_est,label='Simulation Error')                            # Plot Error between model and true system (its almost zero)
plt.legend()
plt.show()


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
