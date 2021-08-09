# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:18:31 2021

@author: alexa
"""
# from sys import path
# path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

import scipy.io

from models.NN import SilverBoxPhysikal


# from controllers import LPV_Controller_full

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

train = SNLS80mV.iloc[40580:49270][['u','y']]-SNLS80mV.mean()
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


model = SilverBoxPhysikal(name='PhysikalSilverboxModel')

identification_results = pkl.load(open('SilverboxPhysical.pkl','rb'))

model.Parameters = identification_results.loc[0,'params']



x=[]
y=[]

x.append(np.zeros((2,1)))

theta = []

for k in range(0,test_u[0].shape[0]):
    
    x_new, y_est = model.OneStepPrediction([0,0],test_u[0,k,:])   
    
    x_new, y_est = np.array(x_new), np.array(y_est)
    
    theta_new = model.EvalAffineParameters(x[-1],test_u[0,k,:])    
    
    
    x.append(x_new)
    y.append(y_est)
    theta.append(np.array(theta_new))
    
theta = np.vstack(theta)
y = np.vstack(y)

plt.scatter(y,theta)

plt.plot(y,theta)


# Get vertices from data
v1 = tuple(-0.00011606)
v2 = tuple(0.00)
# Get vertice systems
S1 = model.AffineStateSpaceMatrices(np.sqrt(-v1),np.array([[0]]))
S2 = model.AffineStateSpaceMatrices(np.sqrt(-v2),np.array([[0]]))


VertexSystems = dict(S1=S1,S2=S2)

scipy.io.savemat('../Matlab_Hinf_Synthesis/Silverbox/VertexSystemsSilverboxPhys.mat',
                 VertexSystems)



















