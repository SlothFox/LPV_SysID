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

from scipy.io import loadmat

from models.RealWorldSystems import RobotManipulator
from testsignals.testsignals import APRBS



''' Design Input Signals '''
N = 1000

# u1 = np.zeros((1,N))#np.hstack((np.zeros((1,20)),10*np.ones((1,80))))

# u2 = np.zeros((1,N))


# # u1 = APRBS(N,[-1,1],[150,250])
# # u2 = APRBS(N,[-1,1],[150,250])

# # u1[0,0:100] = -1
# u1[0,400::] = 0.2
# u = np.vstack((u1,u2)).T

x0 = np.array([[0],[0],[0],[0]])

model = RobotManipulator('RobotManipulator')

x = []
y = []

x.append(x0)

u = np.zeros((1,2))
holding_time = np.zeros((1,2))

holding_range = np.array([[10,50],[50,100]])
step_range = np.array([[-2,2],[-1,1]])

u[0,0] = np.random.rand(1,1) * (step_range[0,1]-step_range[0,0]) + step_range[0,0]
u[0,1] = np.random.rand(1,1) * (step_range[1,1]-step_range[1,0]) + step_range[1,0]

holding_time[0,0] = np.random.rand(1,1) * (holding_range[0,1]-holding_range[0,0]) + holding_range[0,0]
holding_time[0,1] = np.random.rand(1,1) * (holding_range[1,1]-holding_range[1,0]) + holding_range[1,0]


 
# Online Testsignal Generation
for k in range(0,N):
    
    x_new,y_new = model.OneStepPrediction(x[-1],u[-1])
    
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    
    x.append(x_new.T)
    y.append(y_new.T)
    
    holding_time = holding_time -1
    
    # Check if angles are in desired range
    if y_new [0] < -1.9:
        u[0,0] = abs(np.random.rand(1,1)) * (step_range[0,1]-step_range[0,0]) + step_range[0,0]
        holding_time[0,0] = 0

    elif y_new [0] > 1.9:    
        u[0,0] = -abs(np.random.rand(1,1)) * (step_range[0,1]-step_range[0,0]) + step_range[0,0]                                  
        holding_time[0,0] = 0
        
    if y_new [1] < -1.9:
        u[0,1] = abs(np.random.rand(1,1)) * (step_range[0,1]-step_range[0,0]) + step_range[0,0]
        holding_time[0,0] = 0
        
    elif y_new [1] > 1.9:    
        u[0,1] = -abs(np.random.rand(1,1)) * (step_range[0,1]-step_range[0,0]) + step_range[0,0]                                       
        holding_time[0,0] = 0
        
    # Check if holding time of current step is over
    if holding_time[0,0]==0:
        holding_time[0,0] = np.random.rand(1,1) * (holding_range[0,1]-holding_range[0,0]) + holding_range[0,0]
    elif holding_time[0,1]==0:
        holding_time[0,1] = np.random.rand(1,1) * (holding_range[1,1]-holding_range[1,0]) + holding_range[1,0]
    
    holding_time = holding_time.astype(int)


  
plt

