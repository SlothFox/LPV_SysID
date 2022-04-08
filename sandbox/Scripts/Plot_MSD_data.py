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
from optim.common import BestFitRate


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


''' Load best models '''

path = 'Results/MSD/'
Lach = 'MSD_Lachhab_3states_NOE2_lam0.01.pkl'
RBF = 'MSD_RBF_3states.pkl'
LPVNN_NOE = 'LPVNN_NOE_final.pkl'
LPVNN_init = 'MSD_LPVNN_3states_lam0.01.pkl'


BFR_lin = 71.46                     # BFR linear model on test dat

Lach=pkl.load(open(path+Lach,'rb'))
RBF=pkl.load(open(path+RBF,'rb'))
LPVNN_NOE=pkl.load(open(path+LPVNN_NOE,'rb'))
LPVNN_init=pkl.load(open(path+LPVNN_init,'rb'))


# Pick best from LPVNN
LPVNN_NOE = LPVNN_NOE.sort_values('BFR_test',ascending=False).iloc[0:2]
LPVNN_init = LPVNN_init.sort_values('BFR_test',ascending=False).iloc[0:9]

LPVNN_NOE = LPVNN_NOE.append(LPVNN_init)

# Pick best from Lach
Lach = Lach.sort_values('BFR_test',ascending=False).iloc[8:19]
Lach['dim_theta']=1

model_LPV  = NN.Rehmer_NN_LPV(dim_u=1,dim_x=dim_x,dim_y=1,
                        dim_thetaA=1, NN_A_dim=[[5,5,5,1]],
                                         NN_A_act=[[1,1,1,0]]) 
model_LPV.Parameters = LPVNN_NOE.iloc[2]['params']

model_Lach = NN.LachhabLPV(dim_u=1,dim_x=dim_x,dim_y=1, dim_thetaA=1)
model_Lach.Parameters =  Lach.iloc[0]['params']
        
model_RBF = NN.RBFLPV(dim_u=1,dim_x=dim_x,dim_y=1, dim_theta=2)
model_RBF.Parameters =  RBF.iloc[11]['params']

x_LPV,y_LPV = model_LPV.Simulation(data['init_state_train'][0], data['u_test'][0])
x_Lach,y_Lach = model_Lach.Simulation(data['init_state_train'][0], data['u_test'][0])
x_RBF,y_RBF = model_RBF.Simulation(data['init_state_train'][0], data['u_test'][0])

######## Plot Shit ###########################################################
plt.close('all')
palette = sns.color_palette()[1::]

fig, axs = plt.subplots(2,1,figsize=(9/2.54,5/2.54), gridspec_kw={'height_ratios': [2, 1]})

axs[0].plot(data['y_test'][0], linewidth=2)
axs[0].plot(np.array(y_LPV),color=palette[2],label='dLPV-RNN',linewidth=1)
# axs[0].plot(np.array(y_Lach),color=palette[1],label='S-RNN')
axs[0].plot(np.array(y_RBF),color=palette[1],label='RBF-RNN',linewidth=1,linestyle='--')

e_LPV = np.abs(data['y_test'][0]-np.array(y_LPV))
e_RBF = np.abs(data['y_test'][0]-np.array(y_RBF))

axs[1].plot(e_LPV,color=palette[2],linewidth=1)
# axs[1].plot(np.abs(np.array(y_Lach)-data['y_test'][0]),color=palette[1])
axs[1].plot(e_RBF,color=palette[1],linewidth=1,linestyle='--')

axs[1].set_xlabel(r'$k$')
axs[0].set_ylabel(r'$y$')
axs[1].set_ylabel(r'$|y-\hat{y}|$')


axs[0].set_xlim([2300,3800])
axs[1].set_xlim([2300,3800])

axs[1].set_ylim([0,0.0015])

axs[1].set_yticks([0,0.001])
# fig.legend()


fig.tight_layout()
fig.savefig('MSD_NOE_simulation.png', bbox_inches='tight',dpi=600)

















# Load inital linear state space model
LSS=loadmat("Benchmarks/Mass_Spring_Damper_LuGre/data/LuGre_LSS_s3")
LSS=LSS['LuGre_LSS']

SubspaceModel = NN.LinearSSM(dim_u=1,dim_x=dim_x,dim_y=1)
SubspaceModel.Parameters = {'A': LSS[0][0][0],
                  'B': LSS[0][0][1],
                  'C': LSS[0][0][2]}

x,y = SubspaceModel.Simulation(data['init_state_test'][0], data['u_test'][0])

BFR = BestFitRate(data['y_test'][0],np.array(y))

    
# model1 = NN.RBFLPV(dim_u=1,dim_x=2,dim_y=1,dim_theta=3,
#                       initial_params=initial_params,name='RBF_network')


# res1=pkl.load(open('results_statesched/Silverbox_RBF_2states_theta3.pkl','rb'))
# model1.Parameters = res1.loc[8,'params']
# x_est, y_est1 = model1.Simulation(init_state[0],test_u[0])
# y_est1 = np.array(y_est1)



# model2 = NN.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=1,dim_thetaB=0,
#                       dim_thetaC=0, NN_1_dim=[1],NN_2_dim=[],
#                       NN_3_dim=[],NN1_act=[1],NN2_act=[],NN3_act=[], 
#                       initial_params=initial_params,name='Rehmer_LPV')

# res2=pkl.load(open('results_statesched/Silverbox_Rehmer_2states_theta1.pkl','rb'))
# model2.Parameters = res2.loc[8,'params']
# x_est, y_est2 = model2.Simulation(init_state[0],test_u[0])
# y_est2 = np.array(y_est2)



# model3 = NN.LachhabLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=3,dim_thetaB=0,
#                       dim_thetaC=0, initial_params=initial_params,
#                       name='Lachhab_LPV')

# res3=pkl.load(open('results_statesched/Silverbox_Lachhab_2states_theta3.pkl','rb'))
# model3.Parameters = res3.loc[9,'params']
# x_est, y_est3 = model3.Simulation(init_state[0],test_u[0])
# y_est3 = np.array(y_est3)


''' Plot measured Input-Output data'''
# plt.rc(usetex = True)

# params = {'tex.usetex': True}
# plt.rcParams.update(params)

# plt.close('all')

# fig, axs = plt.subplots(2)
# fig.set_size_inches((9/2.54,7/2.54))
# # axs[0].legend(loc='upper right',shadow=False,fancybox=False,frameon=False)


# axs[0].plot(test_u[0],label = '$u$',linewidth=2)
# axs[0].set_xticklabels({})
# axs[0].set_ylim((-0.033,0.033))
# axs[0].set_xlim((0,800))
# axs[0].set_ylabel('$u$')

# axs[1].plot(test_y[0],label = '$y$',linewidth=2)
# axs[1].set_xticklabels({})
# axs[1].set_ylim((-0.0032,0.002))
# axs[1].set_xlim((0,800))
# axs[1].set_ylabel('$y$')
# axs[1].set_xlabel('$k$')

# fig.savefig('TestDataMSD.png', bbox_inches='tight',dpi=600)

# axs[1].plot(test_y[0]-y_est1,'--',label = '$e$',linewidth=1,color=sns.color_palette()[2])
# axs[1].plot(test_y[0]-y_est2,':',label = '$e$',linewidth=1,color=sns.color_palette()[3])
# axs[1].plot(test_y[0]-y_est3,'-.',label = '$e$',linewidth=1,color=sns.color_palette()[1])

# axs[1].set_ylim((-0.025,0.025))
# axs[1].set_xlim((200,500))
# axs[1].set_ylabel('$e$')
# axs[1].set_xlabel('$k$')


# axs[0].plot(test_y[0],label = '$y$',linewidth=2)
# axs[0].plot(y_est1,'--',label = '$\hat{y}$',linewidth=1,color=sns.color_palette()[2])
# axs[0].plot(y_est2,':',label = '$\hat{y}$',linewidth=1,color=sns.color_palette()[3])
# axs[0].plot(y_est3,'-.',label = '$\hat{y}$',linewidth=1,color=sns.color_palette()[1])
# axs[0].set_xticklabels({})
# axs[0].set_ylim((-0.35,0.35))
# axs[0].set_xlim((200,500))
# axs[0].set_ylabel('$y$')


# axs[1].plot(test_y[0]-y_est1,'--',label = '$e$',linewidth=1,color=sns.color_palette()[2])
# axs[1].plot(test_y[0]-y_est2,':',label = '$e$',linewidth=1,color=sns.color_palette()[3])
# axs[1].plot(test_y[0]-y_est3,'-.',label = '$e$',linewidth=1,color=sns.color_palette()[1])

# axs[1].set_ylim((-0.025,0.025))
# axs[1].set_xlim((200,500))
# axs[1].set_ylabel('$e$')
# axs[1].set_xlabel('$k$')
# axs[1].legend(loc='upper left',shadow=False,fancybox=False,frameon=False)

# fig.savefig('Silverbox_TestSimulation.png', bbox_inches='tight',dpi=600)

# fig, axs = plt.subplots(2)
# axs[0].plot(val_y[0],label = '$y$')
# axs[0].plot(val_y[0],label = '$y$')
# axs[0].plot(y_est,label = '$\hat{y}$')
# plt.legend()

# axs[1].plot(val_y[0]-y_est,'g',label = '$e$')
# plt.legend()





    
    