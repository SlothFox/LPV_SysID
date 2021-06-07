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

from models.NN import TestLPV
from testsignals.testsignals import APRBS



model = TestLPV(name='LPVTest')

x0 = np.array([[0],[0]])
u0 = np.array([[0]])

N = 2000

test_u = APRBS(N,[-0.5,0.5],[75,100]).T

x=[]
y=[]

x2=[]
y2=[]


x.append(np.zeros((2,1)))
x2.append(np.zeros((2,1)))

theta = []

for k in range(0,N):
    
    x_new, y_est = model.OneStepPrediction(x[k],test_u[k,:])   
    x_new, y_est = np.array(x_new), np.array(y_est)
    
    theta_new = model.EvalAffineParameters(x2[k], test_u[k,:])
    theta_new = np.array(theta_new)[0][0]
    
    A,B,C = model.AffineStateSpaceMatrices(theta_new)
    x_new2 = np.matmul(A,x2[k]) + np.matmul(B,test_u[[k],:])
    y_new2 = np.matmul(C,x_new2)
    
    # theta_new = model.EvalAffineParameters(x[-1],test_u[0,k,:])    
    
    
    x.append(x_new)
    y.append(y_est)
    
    x2.append(x_new2)
    y2.append(y_new2)
    theta.append(theta_new)
    
# theta = np.vstack(theta)
y = np.hstack(y)
y2 = np.hstack(y2)
# plt.scatter(y,theta)

plt.plot(y[0,:])
plt.plot(y2[0,:])
plt.plot(test_u)


# Get vertices from data
v1 = -0.00
v2 = -0.5
# Get vertice systems
A1,B1,C1 = model.AffineStateSpaceMatrices(v1)
A2,B2,C2 = model.AffineStateSpaceMatrices(v2)

VertexSystems = dict(A1=A1,B1=B1,C1=C1,A2=A2,B2=B2,C2=C2)
scipy.io.savemat('../Matlab_Hinf_Synthesis/Silverbox/VertexSystemsTestLPV.mat',
                  VertexSystems)



















