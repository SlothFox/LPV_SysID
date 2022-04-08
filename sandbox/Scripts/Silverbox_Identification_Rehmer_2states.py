# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim
# from miscellaneous import *

from sklearn.preprocessing import MinMaxScaler


''' Data Preprocessing '''

################ Load Data ####################################################
SNLS80mV = pkl.load(open('Benchmarks/Silverbox/SNLS80mV.pkl','rb'))
Schroeder80mV = pkl.load(open('Benchmarks/Silverbox/Schroeder80mV.pkl','rb'))

################# Scale Data ##################################################
                                          
# scaler = MinMaxScaler(feature_range=(-1,1))

# Validierungsdatensatz2 (Data_val) hat den größten Wertebereich, daher dieses Signal für Skalierung verwenden
# SNLS80mV = pd.DataFrame(data = scaler.fit_transform(SNLS80mV),
#                                   columns=SNLS80mV.columns)
# Schroeder80mV = pd.DataFrame(data = scaler.transform(Schroeder80mV),
#                                     columns=Schroeder80mV.columns)

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


''' Identification '''


# Load inital linear state space model
LSS=loadmat("./Benchmarks/Silverbox/SilverBox_LSS")
LSS=LSS['Results']


''' Approach Lachhab '''
initial_params = {'A_0': LSS['A'][0][0],
                  'B_0': LSS['B'][0][0],
                  'C_0': LSS['C'][0][0]}

p_opts = None #{"expand":False}
s_opts = None #{"max_iter": 1000, "print_level":0, 'hessian_approximation': 'limited-memory'}


''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''

for dim in [1,2,3,4,5]:
    
    model = NN.RehmerLPV_outputSched(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=dim,dim_thetaB=0,
                          dim_thetaC=0, NN_1_dim=[dim],NN_2_dim=[],
                          NN_3_dim=[],NN1_act=[1],NN2_act=[],NN3_act=[], 
                          initial_params=initial_params,name='Rehmer_LPV')


    identification_results = param_optim.ModelTraining(model,data,10,
                             initial_params=initial_params,p_opts=p_opts,
                             s_opts=s_opts)

    pkl.dump(identification_results,open('SilverBox_Rehmer_outSched_2states_theta'+str(dim)+'.pkl',
                                          'wb'))
    
    
