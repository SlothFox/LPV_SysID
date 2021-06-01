# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:37:02 2021

@author: alexa
"""
# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim


def BestFitRate(y_target,y_est):
    BFR = 1-sum((y_target-y_est)**2) / sum((y_target-np.mean(y_target))**2) 
    
    BFR = BFR*100
    
    if BFR<0:
        BFR = np.array([0])
        
    return BFR



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


BFR_on_val_data = pd.DataFrame(columns = ['BFR','model','initialization','theta'])

''' Evaluate models on data, save BFR '''

# models = ['Rehmer']
models = ['Lachhab','RBF','Rehmer']
# models = ['Lachhab','RBF']

dims = [1,2,3,4]

for model in models:
    for dim in dims:
        
        # Load identification results
        file = 'results_statesched/' + 'Bioreactor_' + model + '_2states_theta' + str(dim) + '.pkl'
        ident_results = pkl.load(open(file,'rb'))
                
        
        # Initialize model structure
        if model == 'RBF':
            LPV_model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=dim,
                      initial_params=None,name='RBF_network')  
        elif model == 'Rehmer':
            LPV_model = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,dim_thetaB=dim,
                          dim_thetaC=0, NN_1_dim=[5,dim],NN_2_dim=[dim],
                          NN_3_dim=[],NN1_act=[0,1],NN2_act=[0,1],NN3_act=[], 
                          initial_params={},name='Rehmer_LPV')
        elif model == 'Lachhab':
            LPV_model = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,
                                      dim_thetaB=dim, dim_thetaC=0,
                                      initial_params={},name='Lachhab_LPV') 
        
        # Assign parameters to model and evaluate model fit
        for init in range(0,10):
            LPV_model.Parameters = ident_results.loc[init,'params']
            
            x_est, y_est = LPV_model.Simulation(init_state[0],val_u[0])
            y_est = np.array(y_est)
        
            BFR = BestFitRate(val_y[0],y_est)
            
            new_row = pd.DataFrame(data = [[BFR[0], model, init, dim]], 
                                   columns = ['BFR','model','initialization','theta'])
            
            BFR_on_val_data = BFR_on_val_data.append(new_row, ignore_index=True)

# add one or two RBFs manually
# model='RBF'
# dim = 6
# file = 'results_statesched/' + 'Bioreactor_' + model + '_2states_theta' + str(dim) + '.pkl'
# ident_results = pkl.load(open(file,'rb'))
# LPV_model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=dim,
#                       initial_params=None,name='RBF_network')  

# for init in range(0,10):
#     LPV_model.Parameters = ident_results.loc[init,'params']
    
#     x_est, y_est = LPV_model.Simulation(init_state[0],val_u[0])
#     y_est = np.array(y_est)

#     BFR = BestFitRate(val_y[0],y_est)
    
#     new_row = pd.DataFrame(data = [[BFR[0], model, init, dim]], 
#                            columns = ['BFR','model','initialization','theta'])
    
#     BFR_on_val_data = BFR_on_val_data.append(new_row, ignore_index=True)
            
            
            
# dim theta for lachhab and rehmer is actually dim theta *2 :
BFR_on_val_data.loc[np.arange(0,40,1),'theta'] = BFR_on_val_data.loc[np.arange(0,40,1),'theta']*2
BFR_on_val_data.loc[np.arange(80,120,1),'theta'] = BFR_on_val_data.loc[np.arange(80,120,1),'theta']*2


palette = sns.color_palette()[1::]

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})

fig.set_size_inches((9/2.54,4/2.54))

sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
                  palette=palette, fliersize=2,ax=axs, linewidth=1)

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette="Set1",fliersize=2,ax=axs[1], linewidth=1)

axs.legend_.remove()

axs.set_xlabel(r'$\dim(\theta_k)$')


axs.set_ylabel(None)


axs.set_ylim(-5,100)
# axs.set_xlim(-0.5,3.5)

fig.savefig('Bioreactor_StateSched_Boxplot.png', bbox_inches='tight',dpi=600)

# plt.figure

# ''' Plot measured Input-Output data'''
# # plt.rc(usetex = True)

# # params = {'tex.usetex': True}
# # plt.rcParams.update(params)

# fig, axs = plt.subplots(2)
# fig.set_size_inches((9/2.54,7/2.54))
# axs[0].plot(test_u[0],label = '$u$',linewidth=1)
# axs[0].set_xticklabels({})
# axs[0].set_ylim((-0.05,0.05))
# axs[0].set_xlim((200,500))
# axs[0].set_ylabel('$u$')
# # axs[0].legend(loc='upper right',shadow=False,fancybox=False,frameon=False)

# axs[1].plot(test_y[0],label = '$y$',linewidth=2)
# axs[1].plot(y_est,'--',label = '$\hat{y}$',linewidth=1)
# axs[1].set_ylim((-0.35,0.35))
# axs[1].set_xlim((200,500))
# axs[1].set_ylabel('$y$')
# axs[1].set_xlabel('$k$')
# # axs[1].legend(loc='upper left',shadow=False,fancybox=False,frameon=False)

# fig.savefig('Silverbox_test.png', bbox_inches='tight',dpi=600)