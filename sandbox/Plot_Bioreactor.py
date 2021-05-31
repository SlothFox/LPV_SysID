# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:04:07 2021

@author: alexa
"""

# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim


def BestFitRate(y_target,y_est):
    BFR = 1-sum((y_target-y_est)**2) / sum((y_target-np.mean(y_target))**2) 
    
    BFR = BFR*100
    
    if BFR<0:
        BFR = 0
        
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

""" Choose best model of each approach and simulate over test data """
    
model1 = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=2,
                      initial_params=None,name='RBF_network')
res1=pkl.load(open('results_statesched/Bioreactor_RBF_2states_theta2.pkl','rb'))
model1.Parameters = res1.loc[7,'params']
x_est, y_est1 = model1.Simulation(init_state[0],val_u[0])
y_est1 = np.array(y_est1)



model2 = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,dim_thetaB=1,
                      dim_thetaC=0, NN_1_dim=[5,1],NN_2_dim=[1],
                      NN_3_dim=[],NN1_act=[0,1],NN2_act=[0],NN3_act=[], 
                      initial_params=None,name='Rehmer_LPV')

res2=pkl.load(open('results_statesched/Bioreactor_Rehmer_2states_theta1.pkl','rb'))
model2.Parameters = res2.loc[6,'params']
x_est, y_est2 = model2.Simulation(init_state[0],val_u[0])
y_est2 = np.array(y_est2)



model3 = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=4,dim_thetaB=4,
                      dim_thetaC=0, initial_params=None,
                      name='Lachhab_LPV')

res3=pkl.load(open('results_statesched/Bioreactor_Lachhab_2states_theta4.pkl','rb'))
model3.Parameters = res3.loc[6,'params']
x_est, y_est3 = model3.Simulation(init_state[0],val_u[0])
y_est3 = np.array(y_est3)


''' Plot measured Input-Output data'''
# plt.rc(usetex = True)

# params = {'tex.usetex': True}
# plt.rcParams.update(params)

fig, axs = plt.subplots(2)
fig.set_size_inches((9/2.54,7/2.54))
# axs[0].legend(loc='upper right',shadow=False,fancybox=False,frameon=False)

axs[0].plot(val_y[0],label = '$y$',linewidth=2)
axs[0].plot(y_est1,'--',label = '$\hat{y}$',linewidth=1,color=sns.color_palette()[2])
axs[0].plot(y_est2,':',label = '$\hat{y}$',linewidth=1,color=sns.color_palette()[3])
axs[0].plot(y_est3,'-.',label = '$\hat{y}$',linewidth=1,color=sns.color_palette()[1])
axs[0].set_xticklabels({})
axs[0].set_ylim((0.03,0.16))
axs[0].set_xlim((0,500))
axs[0].set_ylabel('$y$')


axs[1].plot(val_y[0]-y_est1,'--',label = '$e$',linewidth=1,color=sns.color_palette()[2])
axs[1].plot(val_y[0]-y_est2,':',label = '$e$',linewidth=1,color=sns.color_palette()[3])
axs[1].plot(val_y[0]-y_est3,'-.',label = '$e$',linewidth=1,color=sns.color_palette()[1])

axs[1].set_ylim((-0.025,0.025))
axs[1].set_xlim((0,500))
axs[1].set_ylabel('$e$')
axs[1].set_xlabel('$k$')
# axs[1].legend(loc='upper left',shadow=False,fancybox=False,frameon=False)

fig.savefig('Bioreactor_TestSimulation.png', bbox_inches='tight',dpi=600)
    
    