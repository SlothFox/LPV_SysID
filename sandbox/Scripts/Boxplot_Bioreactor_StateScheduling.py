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

import sys
sys.path.insert(0,'E:\GitHub\LPV_SysID\sandbox')

from models import NN
from optim import param_optim




def BestFitRate(y_target,y_est):
    BFR = 1-sum((y_target-y_est)**2) / sum((y_target-np.mean(y_target))**2) 
    
    BFR = BFR*100
    
    if BFR<0:
        BFR = np.array([0])
        
    return BFR



''' Data Preprocessing '''

################ Load Data ####################################################
train = loadmat('../Benchmarks/Bioreactor/APRBS_Data_3')
train = train['data']
val = loadmat('../Benchmarks/Bioreactor/APRBS_Data_1')
val = val['data']
test = loadmat('../Benchmarks/Bioreactor/APRBS_Data_2')
test = test['data']


################ Subsample Data ###############################################
train = pd.DataFrame(data=train[0::50,:],columns = ['u','y'])
val = pd.DataFrame(data=val[0::50,:],columns = ['u','y'])
test = pd.DataFrame(data=test[0::50,:],columns = ['u','y'])
################# Pick Training- Validation- and Test-Data ####################

init_state = np.zeros((1,2,1))

# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'data':[val], 'init_state':[init_state]}


BFR_on_val_data = pd.DataFrame(columns = ['BFR','model','initialization','theta'])

''' Evaluate models on data, save BFR '''

# models = ['Rehmer']
models = ['Lachhab','RBF','Rehmer']
# models = ['Lachhab','RBF']

dims = [1,2,3,4]

for model in models:
    for dim in dims:
        
        # Load identification results
        file = '../results_statesched/' + 'Bioreactor_' + model + '_2states_theta' + str(dim) + '.pkl'
        ident_results = pkl.load(open(file,'rb'))
                
        
        # Initialize model structure
        if model == 'RBF':
            LPV_model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=dim,
                      initial_params=None,init_proc='xavier')  
        elif model == 'Rehmer':
            LPV_model = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,dim_thetaB=dim,
                          dim_thetaC=0, NN_1_dim=[5,dim],NN_2_dim=[dim],
                          NN_3_dim=[],NN1_act=[1,2],NN2_act=[1,2],NN3_act=[], 
                          initial_params={})
        elif model == 'Lachhab':
            LPV_model = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,
                                      dim_thetaB=dim, dim_thetaC=0,
                                      initial_params={}) 
        
        # Assign parameters to model and evaluate model fit
        for init in range(0,10):
            LPV_model.Parameters = ident_results.loc[init,'params']
            _,sim = param_optim.parallel_mode(LPV_model,data)
            
            y_est = sim['y']

       
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

BFR_on_val_data['BFR']=BFR_on_val_data['BFR'].astype('float64')


palette = sns.color_palette()[1::]

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})

fig.set_size_inches((9/2.54,4/2.54))

# sns.violinplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette=palette, fliersize=2,ax=axs, linewidth=1)
sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, ax=axs,
               color=".8")
sns.stripplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
                  palette=palette, ax=axs, linewidth=0.1,
                  dodge=True,zorder=1)

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette="Set1",fliersize=2,ax=axs[1], linewidth=1)

axs.legend_.remove()

axs.set_xlabel(r'$\dim(\theta_k)$')

axs.set_ylabel(None)


axs.set_ylim(-5,100)

fig.savefig('Bioreactor_StateSched_Boxplot.png', bbox_inches='tight',dpi=600)

