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

train = pd.DataFrame(data=train,columns=['u','y'])
val = pd.DataFrame(data=val,columns=['u','y'])



init_state = np.zeros((2,1))

# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data_train = {'data':[train], 'init_state': [init_state]}
data_val = {'data':[val], 'init_state': [init_state]}

''' Identification '''
# # Load inital linear state space model
# LSS=loadmat("./Benchmarks/Bioreactor/Bioreactor_LSS")
# LSS=LSS['Results']


# ''' RBF approach'''

# initial_params = {'A': LSS['A'][0][0],
#                   'B': LSS['B'][0][0],
#                   'C': LSS['C'][0][0],
#                   'range_u': np.array([[0,0.7]]),
#                   'range_x': np.array([[-0.1,0.1],[-0.1,0.1]])}

# p_opts = {"expand":False}
# s_opts = {"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}


''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''

for dim in [6,8]:
    
    model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,u_lab=['u'],y_lab=['y'],
                      dim_theta=dim,initial_params=None)    
    
    identification_results = param_optim.ModelTraining(model,data_train,
                                data_val,5,p_opts=None,s_opts=None, mode='parallel')

    pkl.dump(identification_results,open('Home_Bioreactor_RBF_2states_theta'+str(dim)+'.pkl',
                                         'wb'))


