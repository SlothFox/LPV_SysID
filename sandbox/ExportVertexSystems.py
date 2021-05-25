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

from models.NN import RehmerLPV

from sklearn.preprocessing import MinMaxScaler

# from controllers import LPV_Controller_full

''' Data Preprocessing '''

################ Load Data ####################################################
SNLS80mV = pkl.load(open('Benchmarks/Silverbox/SNLS80mV.pkl','rb'))
Schroeder80mV = pkl.load(open('Benchmarks/Silverbox/Schroeder80mV.pkl','rb'))

################# Scale Data ##################################################
                                          
scaler = MinMaxScaler(feature_range=(-1,1))

# Validierungsdatensatz2 (Data_val) hat den größten Wertebereich, daher dieses Signal für Skalierung verwenden
SNLS80mV = pd.DataFrame(data = scaler.fit_transform(SNLS80mV),
                                  columns=SNLS80mV.columns)
Schroeder80mV = pd.DataFrame(data = scaler.transform(Schroeder80mV),
                                    columns=Schroeder80mV.columns)

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



model = RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=2,dim_thetaB=0,
                          dim_thetaC=0,fA_dim=2,fB_dim=0,fC_dim=0,
                          initial_params=None,name='name')

identification_results = pkl.load(open('Benchmarks/Silverbox/IdentifiedModels/Silverbox_Topmodel.pkl','rb'))

model.Parameters = identification_results.loc[0,'params']


x_est, y_est = model.Simulation([0,0],test_u[0])

y_est = np.array(y_est)

plt.plot(y_est,label='Estimation')
plt.plot(test_y[0], label='TrueOutput')
plt.plot(test_y[0]-y_est, label='error')
plt.legend()

x=[]
y=[]

x.append(np.zeros((2,1)))

theta = []

for k in range(0,test_u[0].shape[0]):
    
    x_new, y_est = model.OneStepPrediction([0,0],test_u[0,k,:])    
    theta_new = model.EvalAffineParameters(x[-1],test_u[0,k,:])    
    
    
    x.append(x_new)
    theta.append(theta_new)
    



# Get vertices from data
# v1 = (0.51,0.61)
# v2 = (0.51,0.65)
# v3 = (0.515,0.65)
# v4 = (0.515,0.61)

# Get vertice systems
# S1 = model.AffineStateSpaceMatrices(v1)
# S2 = model.AffineStateSpaceMatrices(v2)
# S3 = model.AffineStateSpaceMatrices(v3)
# S4 = model.AffineStateSpaceMatrices(v4)

# VertexSystems = dict(S1=S1,S2=S2,S3=S3,S4=S4)

# scipy.io.savemat('VertexSystemsSilverbox.mat', VertexSystems)

# LPV_Controller = LPV_Controller_full(Omega=None, vertices=(v1,v2,v3,v4))

# x = np.zeros((2,1))
# y = 


















