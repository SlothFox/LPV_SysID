# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:32:58 2021

@author: alexa
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.io import loadmat, savemat

from models.RealWorldSystems import RobotManipulator,RobotManipulator2
from testsignals.testsignals import APRBS



''' Design Input Signals '''
N = 5000



x0 = np.array([[-np.pi/2],[0],[0],[0]])

model = RobotManipulator2('RobotManipulator')


x = []
y = []
u = []

x.append(x0)

u.append(np.zeros((1,2)))
holding_time = np.zeros((1,2))

holding_range = np.array([[200,300],[100,120]])
step_range = np.array([[-1,5],[-0.8,0.9]])

u[0][0,0] = np.random.rand(1,1) * (step_range[0,1]-step_range[0,0]) + step_range[0,0]
u[0][0,1] = np.random.rand(1,1) * (step_range[1,1]-step_range[1,0]) + step_range[1,0]

holding_time[0,0] = abs(np.random.rand(1,1) * (holding_range[0,1]-holding_range[0,0]) + holding_range[0,0])
holding_time[0,1] = abs(np.random.rand(1,1) * (holding_range[1,1]-holding_range[1,0]) + holding_range[1,0])


locku1 = False
locku2 = False
 
# Online Testsignal Generation
for k in range(0,N):
    # u[-1]=np.zeros((1,2))
    x_new,y_new = model.OneStepPrediction(x[-1],u[-1])
    # print(y_new)
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    
    x.append(x_new.T)
    y.append(y_new.T)
    
    holding_time = holding_time -1
    # print(holding_time)
    # Check if angles are in desired range
    u_new = np.copy(u[-1])
    
    if y_new [0] < -np.pi/2 and locku1==False:
        u_new[0,0] = step_range[0,1] #abs(np.random.rand(1,1)* (step_range[0,1]-step_range[0,0]) + step_range[0,0])
        holding_time[0,0] = abs(np.random.rand(1,1) * (holding_range[0,1]-holding_range[0,0]) + holding_range[0,0])
        locku1 = True

    elif y_new [0] > np.pi/2 and locku1==False:    
        u_new[0,0] = step_range[0,0] #-abs(np.random.rand(1,1) * (step_range[0,1]-step_range[0,0]) + step_range[0,0])                                  
        holding_time[0,0] = abs(np.random.rand(1,1) * (holding_range[0,1]-holding_range[0,0]) + holding_range[0,0])
        locku1 = True
        
    if y_new [1] < -np.pi/2 and locku2==False:    
        u_new[0,1] = abs(np.random.rand(1,1) * (step_range[1,1]-step_range[1,0]) + step_range[1,0])
        holding_time[0,1] = abs(np.random.rand(1,1) * (holding_range[1,1]-holding_range[1,0]) + holding_range[1,0])
        locku2 = True
        
    elif y_new [1] > np.pi/2 and locku2==False:       
        u_new[0,1] = -abs(np.random.rand(1,1) * (step_range[1,1]-step_range[1,0]) + step_range[1,0])                                       
        holding_time[0,1] = abs(np.random.rand(1,1) * (holding_range[1,1]-holding_range[1,0]) + holding_range[1,0])
        locku2 = True
        
    # Check if holding time of current step is over
    if holding_time[0,0]<=0:
        u_new[0,0] = np.random.rand(1,1)* (step_range[0,1]-step_range[0,0]) + step_range[0,0]
        holding_time[0,0] = abs(np.random.rand(1,1) * (holding_range[0,1]-holding_range[0,0]) + holding_range[0,0])
        locku1 = False
        
    elif holding_time[0,1]<=0:
        u_new[0,1] = np.random.rand(1,1) * (step_range[1,1]-step_range[1,0]) + step_range[1,0]
        holding_time[0,1] = abs(np.random.rand(1,1) * (holding_range[1,1]-holding_range[1,0]) + holding_range[1,0])
        locku2 = False
        
    u.append(u_new)
    
    holding_time = holding_time.astype(int)


y = np.vstack(y)  
u = np.vstack(u) 

plt.close('all')

plt.figure()
plt.plot(y)
plt.plot(u)

plt.figure()
plt.scatter(y[:,0],y[:,1])
plt.xlim([-2,2])
plt.ylim([-2,2])


io_data = np.hstack((u[:-1],y))

mdic = {"data": io_data, "label": "APRBS_Val_Data"}

# RobotManipulator2

savemat('Benchmarks/RobotManipulator/APRBS_Ident_Data.mat',mdic)
