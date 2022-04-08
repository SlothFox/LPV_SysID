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
data = {'u_train':train_u, 'y_train':train_y, 'init_state_train': init_state,
        'u_val':val_u , 'y_val':val_y,'init_state_val': init_state,
        'u_test':test_u, 'y_test':test_y,'init_state_test': init_state}


init_results = pkl.load(open('Results/MSD/MSD_LPVNN_3states_lam0.01.pkl','rb'))

best_init_results = init_results.sort_values('BFR_test',ascending=False).iloc[0:30]

s_opts = {"max_iter": 2000}#, "print_level":0, 'hessian_approximation': 'limited-memory'}


for i in range(21,22):
# for i in range(0,2):    
    dim_phi = best_init_results.iloc[i]['dim_phi']
    
    model = NN.Rehmer_NN_LPV(dim_u=1,dim_x=dim_x,dim_y=1,
                        dim_thetaA=1, NN_A_dim=[[5,dim_phi,5,1]],
                                         NN_A_act=[[1,1,1,0]]) 
    
    model.InitialParameters =  best_init_results.iloc[i]['params']

    results_NOE = param_optim.ModelTraining(model,data,1,
                                         p_opts=None,s_opts=s_opts)
    
    # Add information
    results_NOE['dim_phi'] = best_init_results.iloc[i]['dim_phi']
    results_NOE['activations'] = best_init_results.iloc[i]['activations']
    results_NOE['dim_thetaA'] = best_init_results.iloc[i]['dim_thetaA']
    results_NOE['lambda'] = best_init_results.iloc[i]['lambda']
    results_NOE['structure'] = best_init_results.iloc[i]['structure']
    
    pkl.dump(results_NOE,open('Results/MSD/LPVNN_NOE_'+str(i),'wb'))
    
    try:
        results = results.append(results_NOE)
    except NameError:
        results = results_NOE    

   
pkl.dump(results,open('./Results/MSD/MSD_LPVNN_3states_NOE_'+
                                          'lam'+str(0.01)
                                          +'.pkl','wb'))
