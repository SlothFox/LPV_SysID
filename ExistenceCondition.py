# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:18:31 2021

@author: alexa
"""
from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

import Modellklassen as Model

model = Model.RehmerLPV(dim_u=1,dim_x=2,dim_y=1,dim_thetaA=2,dim_thetaB=0,
                          dim_thetaC=0,fA_dim=2,fB_dim=0,fC_dim=0,
                          initial_params=None,name='name')

identification_results = pkl.load(open('Benchmarks/Silverbox/IdentifiedModels/Silverbox_Topmodel.pkl','rb'))

model.Parameters = identification_results.loc[0,'params']

# Get vertices from data
v1 = [0.51,0.61]
v2 = [0.51,0.65]
v3 = [0.515,0.65]
v4 = [0.515,0.61]

# Get vertice systems
S1 = model.AffineStateSpaceMatrices(v1)
S2 = model.AffineStateSpaceMatrices(v2)
S3 = model.AffineStateSpaceMatrices(v3)
S4 = model.AffineStateSpaceMatrices(v4)


# Formulate Optimization Problem
import cvxopt as cvx
from cvxopt import matrix, solvers

c = cvx.matrix([1.])  

cvx.solvers.sdp(c[, Gl, hl[, Gs, hs[, A, b[, solver[, primalstart[, dualstart]]]]]]
                  
c = cvx.matrix([1.])            















