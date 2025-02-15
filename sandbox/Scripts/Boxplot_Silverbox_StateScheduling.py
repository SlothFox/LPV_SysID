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
SNLS80mV = pkl.load(open('Benchmarks/Silverbox/SNLS80mV.pkl','rb'))
Schroeder80mV = pkl.load(open('Benchmarks/Silverbox/Schroeder80mV.pkl','rb'))

################# Pick Training- Validation- and Test-Data ####################

train = SNLS80mV.iloc[40580:41270][['u','y']]-SNLS80mV.mean()       #SNLS80mV.iloc[40580:49270][['u','y']]-SNLS80mV.mean()
val = SNLS80mV.iloc[0:40580][['u','y']]-SNLS80mV.mean()
test = Schroeder80mV.iloc[10585:10585+1023][['u','y']]-Schroeder80mV.mean()

train_u = np.array(train[0:-1]['u']).reshape(1,-1,1)
train_y = np.array(train[1::]['y']).reshape(1,-1,1)

val_u = np.array(val[0:-1]['u']).reshape(1,-1,1)
val_y = np.array(val[1::]['y']).reshape(1,-1,1)

test_u = np.array(test[0:-1]['u']).reshape(1,-1,1)
test_y = np.array(test[1::]['y']).reshape(1,-1,1)


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

dims = [1,2,3,4,5]

for model in models:
    for dim in dims:
        
        # Load identification results
        file = 'results_statesched/' + 'SilverBox_' + model + '_2states_theta' + str(dim) + '.pkl'
        ident_results = pkl.load(open(file,'rb'))
        
        # Initialize model structure
        if model == 'RBF':
            LPV_model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=dim,
                      initial_params=None,init_proc='xavier')  
        elif model == 'Rehmer':
            LPV_model = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,dim_thetaB=0,
                          dim_thetaC=0, NN_1_dim=[dim],NN_2_dim=[],
                          NN_3_dim=[],NN1_act=[2],NN2_act=[],NN3_act=[], 
                          initial_params={})
        elif model == 'Lachhab':
            LPV_model = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,
                                      dim_thetaB=dim, dim_thetaC=0,
                                      initial_params={},name='Lachhab_LPV') 
        
        # Assign parameters to model and evaluate model fit
        for init in range(0,10):
            LPV_model.Parameters = ident_results.loc[init,'params']
            
            x_est, y_est = LPV_model.Simulation(init_state[0],test_u[0])
            y_est = np.array(y_est)
        
            BFR = BestFitRate(test_y[0],y_est)
            
            new_row = pd.DataFrame(data = [[BFR[0], model, init, dim]], 
                                   columns = ['BFR','model','initialization','theta'])
            
            BFR_on_val_data = BFR_on_val_data.append(new_row, ignore_index=True)

BFR_on_val_data['BFR']=BFR_on_val_data['BFR'].astype('float64')

# dim theta for lachhab and rehmer is actually dim theta *2 :
# BFR_on_val_data.loc[np.arange(0,50,1),'theta'] = BFR_on_val_data.loc[np.arange(0,50,1),'theta']*2
# BFR_on_val_data.loc[np.arange(100,150,1),'theta'] = BFR_on_val_data.loc[np.arange(100,150,1),'theta']*2


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


axs.set_ylim(-5,105)


fig.savefig('Silverbox_StateSched_Boxplot.png', bbox_inches='tight',dpi=600)


# Old Plot 

# fig, axs = plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})
# fig.set_size_inches((9/2.54,7/2.54))

# palette = sns.color_palette()[1::]

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette=palette, fliersize=2,ax=axs[0], linewidth=1)

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette=palette,fliersize=2,ax=axs[1], linewidth=1)


# axs[0].legend_.remove()
# axs[1].legend_.remove()

# axs[0].set_xticks([])

# axs[1].set_xlabel(r'$\dim(\theta_k)$')
# axs[0].set_xlabel(None)

# axs[0].set_ylabel(None)
# axs[1].set_ylabel(None)

# axs[1].set_ylim(-5,105)
# axs[0].set_ylim(98,100)

# fig.savefig('Silverbox_StateSched_Boxplot.png', bbox_inches='tight',dpi=600)
