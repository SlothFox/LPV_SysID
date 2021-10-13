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
inits = 10

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



''' Identification '''



# Load inital linear state space model

''' Simulate linear model to get range of state for scheduling '''
# Load inital linear state space model
LSS=loadmat("Benchmarks/Mass_Spring_Damper_LuGre/data/LuGre_LSS_s3")
LSS=LSS['LuGre_LSS']

SubspaceModel = NN.LinearSSM(dim_u=1,dim_x=dim_x,dim_y=1)
SubspaceModel.Parameters = {'A': LSS['A'][0][0],
                  'B': LSS['B'][0][0],
                  'C': LSS['C'][0][0]}

x_lin,_ = SubspaceModel.Simulation(data['init_state_train'][0],data['u_train'][0])

x_max = np.max(x_lin,0).reshape(-1,1)
x_min = np.min(x_lin,0).reshape(-1,1)

u_max = np.max(data['u_train'][0],0)
u_min = np.min(data['u_train'][0],0)



''' Start training '''

dim_theta = [1,2,3,4,5]

counter = 0

# s_opts = {"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}

for dim in dim_theta:

    
    model = NN.RBFLPV(dim_u=1,dim_x=dim_x,dim_y=1,dim_theta=dim)  
    
    initial_params = {}
    
    for i in range(0,dim):
        initial_params['A'+str(i)] =  LSS['A'][0][0]
        initial_params['B'+str(i)] =  LSS['B'][0][0]
        initial_params['C'+str(i)] =  LSS['C'][0][0]
        initial_params['c_u'+str(i)] = u_min + \
                                (u_max-u_min) * np.random.uniform(size=(1,1))
        initial_params['c_x'+str(i)] = x_min + \
                                (x_max-x_min) * np.random.uniform(size=(dim_x,1))        
            
    model.InitialParameters = initial_params
    
    results_new = param_optim.ModelTraining(model,data,inits,
                             p_opts=None,s_opts=None)
            
    # Add information?
                
    # Save results
    pkl.dump(results_new,open('./Results/MSD/MSD_RBF_3states_'+str(counter)+ '.pkl','wb'))
    try:
        results = results.append(results_new)
    except NameError:
        results = results_new
    
    counter = counter + 1
   
pkl.dump(results,open('./Results/MSD/MSD_RBF_3states.pkl','wb'))