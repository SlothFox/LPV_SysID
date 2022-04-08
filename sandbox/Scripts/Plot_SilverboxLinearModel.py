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



# Load inital linear state space model
LSS=loadmat("./Benchmarks/Silverbox/SilverBox_LSS")
LSS=LSS['Results']

initial_params = {'A': LSS['A'][0][0],
                  'B': LSS['B'][0][0],
                  'C': LSS['C'][0][0],
                  'range_u': np.array([[-0.05,0.05]]),
                  'range_x': np.array([[-0.2,0.2],[-0.2,0.2]])}

    
model = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=3,
                      initial_params=initial_params,name='RBF_network')

# model = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,dim_thetaB=1,
#                       dim_thetaC=0, NN_1_dim=[5,1],NN_2_dim=[1],
#                       NN_3_dim=[],NN1_act=[0,1],NN2_act=[0,1],NN3_act=[], 
#                       initial_params=initial_params,name='Rehmer_LPV')

res=pkl.load(open('results_statesched/Silverbox_RBF_2states_theta3.pkl','rb'))

model.Parameters = res.loc[8,'params']

x_est, y_est = model.Simulation(init_state[0],test_u[0])

y_est = np.array(y_est)

plt.figure

''' Plot measured Input-Output data'''
# plt.rc(usetex = True)

# params = {'tex.usetex': True}
# plt.rcParams.update(params)

fig, axs = plt.subplots(2)
fig.set_size_inches((9/2.54,7/2.54))
axs[0].plot(test_u[0],label = '$u$',linewidth=1)
axs[0].set_xticklabels({})
axs[0].set_ylim((-0.05,0.05))
axs[0].set_xlim((200,500))
axs[0].set_ylabel('$u$')
# axs[0].legend(loc='upper right',shadow=False,fancybox=False,frameon=False)

axs[1].plot(test_y[0],label = '$y$',linewidth=2)
axs[1].plot(y_est,'--',label = '$\hat{y}$',linewidth=1)
axs[1].set_ylim((-0.35,0.35))
axs[1].set_xlim((200,500))
axs[1].set_ylabel('$y$')
axs[1].set_xlabel('$k$')
# axs[1].legend(loc='upper left',shadow=False,fancybox=False,frameon=False)

fig.savefig('Silverbox_test.png', bbox_inches='tight',dpi=600)

# fig, axs = plt.subplots(2)
# axs[0].plot(val_y[0],label = '$y$')
# axs[0].plot(val_y[0],label = '$y$')
# axs[0].plot(y_est,label = '$\hat{y}$')
# plt.legend()

# axs[1].plot(val_y[0]-y_est,'g',label = '$e$')
# plt.legend()

BFR = BestFitRate(test_y[0],y_est)



    
    