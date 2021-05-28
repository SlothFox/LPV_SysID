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

# Load inital linear state space model
LSS=loadmat("./Benchmarks/Bioreactor/Bioreactor_LSS")
LSS=LSS['Results']

initial_params = {'A': LSS['A'][0][0],
                  'B': LSS['B'][0][0],
                  'C': LSS['C'][0][0],
                  'range_u': np.array([[0,0.7]]),
                  'range_x': np.array([[0.004,0.231],[-0.248,0.0732]])}

    
model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=1,
                      initial_params=initial_params,name='RBF_network')

# model = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,dim_thetaB=1,
#                       dim_thetaC=0, NN_1_dim=[5,1],NN_2_dim=[1],
#                       NN_3_dim=[],NN1_act=[0,1],NN2_act=[0,1],NN3_act=[], 
#                       initial_params=initial_params,name='Rehmer_LPV')

# res=pkl.load(open('Bioreactor_Rehmer_2states_theta1_tryanderror.pkl','rb'))

# model.Parameters = res.loc[1,'params']

x_est, y_est = model.Simulation(init_state[0],val_u[0])

y_est = np.array(y_est)

plt.figure

''' Plot measured Input-Output data'''
# plt.rc(usetex = True)

# params = {'tex.usetex': True}
# plt.rcParams.update(params)

fig, axs = plt.subplots(2)
fig.set_size_inches((9/2.54,7/2.54))
axs[0].plot(val_u[0],label = '$u$')
axs[0].set_xticklabels({})
axs[0].set_ylim((0,1.0))
axs[0].set_xlim((0,500))
axs[0].set_ylabel('$u$')
# axs[0].legend(loc='upper right',shadow=False,fancybox=False,frameon=False)

axs[1].plot(val_y[0],label = '$y$',linewidth=1)
axs[1].plot(y_est,'--',label = '$\hat{y}$')
axs[1].set_ylim((0.03,0.16))
axs[1].set_xlim((0,500))
axs[1].set_ylabel('$y$')
axs[1].set_xlabel('$k$')
# axs[1].legend(loc='upper left',shadow=False,fancybox=False,frameon=False)

fig.savefig('Bioreactor_val.png', bbox_inches='tight',dpi=600)

# fig, axs = plt.subplots(2)
# axs[0].plot(val_y[0],label = '$y$')
# axs[0].plot(val_y[0],label = '$y$')
# axs[0].plot(y_est,label = '$\hat{y}$')
# plt.legend()

# axs[1].plot(val_y[0]-y_est,'g',label = '$e$')
# plt.legend()

BFR = BestFitRate(val_y[0],y_est)



    
    