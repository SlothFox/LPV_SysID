# -*- coding: utf-8 -*-
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat

import Modellklassen as Model
from OptimizationTools import *
from miscellaneous import *

from sklearn.preprocessing import MinMaxScaler

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

train = SNLS80mV.iloc[40580:49270][['u','y']]-SNLS80mV.mean()
val = SNLS80mV.iloc[0:40580][['u','y']]-SNLS80mV.mean()
test = Schroeder80mV.iloc[10585:10585+1023][['u','y']]-Schroeder80mV.mean()

# train_u = np.array(train[0:-1]['u']).reshape(1,-1,1)
train_y = np.array(train[1::]['y']).reshape(1,-1,1)

# val_u = np.array(val[0:-1]['u']).reshape(1,-1,1)
val_y = np.array(val[1::]['y']).reshape(1,-1,1)

# test_u = np.array(test[0:-1]['u']).reshape(1,-1,1)
test_y = np.array(test[1::]['y']).reshape(1,-1,1)

rate = 10

train_u = np.array(train[0:-1]['u'])
train_u = numpy.interp(np.linspace(0,train_u.shape[0],rate*train_u.shape[0]),np.linspace(0,train_u.shape[0],train_u.shape[0]),train_u)
train_u = train_u.reshape(1,-1,1)

# train_y = np.array(train[1::]['y'])
# train_y = numpy.interp(np.linspace(0,train_y.shape[0],rate*train_y.shape[0]),np.linspace(0,train_y.shape[0],train_y.shape[0]),train_y)
# train_y = train_y.reshape(1,-1,1)

val_u = np.array(val[0:-1]['u'])
val_u = numpy.interp(np.linspace(0,val_u.shape[0],rate*val_u.shape[0]),np.linspace(0,val_u.shape[0],val_u.shape[0]),val_u)
val_u = val_u.reshape(1,-1,1)

# val_y = np.array(val[1::]['y'])
# val_y = numpy.interp(np.linspace(0,val_y.shape[0],rate*val_y.shape[0]),np.linspace(0,val_y.shape[0],val_y.shape[0]),val_y)
# val_y = val_y.reshape(1,-1,1)

test_u = np.array(test[0:-1]['u'])
test_u = numpy.interp(np.linspace(0,test_u.shape[0],rate*test_u.shape[0]),np.linspace(0,test_u.shape[0],test_u.shape[0]),test_u)
test_u = test_u.reshape(1,-1,1)

# test_y = np.array(test[1::]['y'])
# test_y = numpy.interp(np.linspace(0,test_y.shape[0],rate*test_y.shape[0]),np.linspace(0,test_y.shape[0],test_y.shape[0]),test_y)
# test_y = test_y.reshape(1,-1,1)


init_state = np.zeros((1,2,1))


# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys
data = {'u_train':train_u, 'y_train':train_y,'init_state_train': init_state,
        'u_val':val_u, 'y_val':val_y,'init_state_val': init_state}



model = Model.SilverBoxPhysikal(name='PhysikalSilverboxModel')




''' Call the Function ModelTraining, which takes the model and the data and 
starts the optimization procedure 'initializations'-times. '''
identification_results = ModelTraining(model,data,5)

#identification_results = pkl.load(open('Benchmarks/Silverbox/IdentifiedModels/Silverbox_Topmodel.pkl','rb'))

''' The output is a pandas dataframe which contains the results for each of
the 10 initializations, specifically the loss on the validation data
and the estimated parameters ''' 

# Pick the parameters from the second initialization (for example, in this case
# every model has a loss close to zero because the optimizer is really good
# and its 'only' a linear model which we identify)

model.Parameters = identification_results.loc[0,'params']


# test_u[0] = 10*np.ones((1022,1))

# Maybe plot the simulation result to see how good the model performs
y_est = model.Simulation(init_state[0],train_u[0])

y_est = np.array(y_est) 


plt.plot(train_y[0],label='True output')                                        # Plot True data
plt.plot(y_est,label='Est. output')                                            # Plot Model Output
plt.plot(test_y[0]-y_est,label='Simulation Error')                             # Plot Error between model and true system (its almost zero)
plt.legend()
plt.show()

# plt.figure()
# plt.plot(thetaA[:,0],label='Theta_A1')    
# plt.plot(thetaA[:,1],label='Theta_A2')   
# plt.scatter(theta[:,0],theta[:,1])  
# plt.legend()
# plt.show()
# e2 = y[0]-y_est

# model.AffineStateSpaceMatrices([1,1])
