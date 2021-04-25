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

import Modellklassen as Model
from OptimizationTools import *
from miscellaneous import *

''' Design Input Signals '''
N = 1000

u1 = np.zeros((1,N))#np.hstack((np.zeros((1,20)),10*np.ones((1,80))))

u2 = np.zeros((1,N))


u1[0,100:110] = -3
u2[0,100:110] = 0

u = np.vstack((u1,u2)).T

x0 = np.array([[np.pi/2],[0],[0],[0]])

model = Model.RobotManipulator('RobotManipulator')

y = model.Simulation(x0, u)

# plt.plot(u)
plt.plot(y)

