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

from models.NN import RehmerLPV_old

from sklearn.preprocessing import MinMaxScaler

from controllers.controllers import LPV_Controller_full

# Load identified model to use as plant
model = RehmerLPV_old(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=2,dim_thetaB=0,
                          dim_thetaC=0,fA_dim=2,fB_dim=0,fC_dim=0,
                          initial_params=None,name='name')
identification_results = pkl.load(open('Benchmarks/Silverbox/IdentifiedModels/Silverbox_Topmodel.pkl','rb'))
model.Parameters = identification_results.loc[0,'params']

# Load identified controller parameters and arrange vertex controller in list
Omega = scipy.io.loadmat('../Matlab_Hinf_Synthesis/Silverbox/VertexControllers')
Omega = Omega['VertexController']
Omega = [VertexController for VertexController in Omega[0,:]]

# Vertices are hyperbox put around time-varying parameters measured during simulation
v1 = (0.51,0.61)
v2 = (0.51,0.65)
v3 = (0.515,0.65)
v4 = (0.515,0.61)

model.AffineStateSpaceMatrices(v1)

vertices = [v1,v2,v3,v4]

controller = LPV_Controller_full(Omega=Omega, vertices=vertices, x_dim = 2,
                                 y_dim = 1, u_dim = 1) 



controller.PolytopicCoords_Hypercube(v2)








N = 100 # length of experiment

# Define reference signal as simple step at k=0 with height 0.1

w = -np.ones((N,1)) * 0.4

# Arrays for system state, output, input, time-varying parameter
x_p = []
x_c = []
theta = []
u = []
y = []
e = []

x_p.append( np.zeros((2,1)) + 10E-6)    # Add something to prevent division by zero
x_c.append(np.zeros((2,1)) + 10E-6)     # Add something to prevent division by zero
u.append(np.zeros((1,1)) + 10E-6)       # Add something to prevent division by zero

for k in range(0,w.shape[0]):
    
    x_p_new, y_new = model.OneStepPrediction(x_p[k],u[k])    
    x_p_new,y_new = np.array(x_p_new), np.array(y_new)
    
    theta_new = model.EvalAffineParameters(x_p_new,u[k])    
    theta_new = np.array(theta_new)
    
    # e_new = w[k] - y_new 
    e_new = y_new - w[k]
    x_c_new, u_new = controller.CalculateControlInput(tuple(theta_new),e_new,x_c[k])
    x_c_new,u_new = np.array(x_c_new), np.array(u_new)
    
    x_p.append(x_p_new)
    x_c.append(x_c_new)
    theta.append(theta_new)
    u.append(u_new)
    y.append(y_new)
    e.append(e_new)
    

    



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


















