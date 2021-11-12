#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:21:22 2021

@author: alexander
"""
import numpy as np 


from miscellaneous.PreProcessing import (hankel_matrix,extend_observ_matrix,
toeplitz, project_row_space)

from models.NN import LinearSSM

N=1000

dim_u = 2
dim_x = 2
dim_y = 2

LSS = LinearSSM(dim_u=dim_u,dim_x=dim_x,dim_y=dim_y)

LSS.Parameters = {'A': np.array([[0.9, 0.5],
                                 [0.5, 0.8]]),
                  'B': np.array([[1, 0],
                                [0, 1]]),
                  'C': np.array([[1, 0],
                                [0, 1]])}  


u = np.random.randn(N,dim_u)
x,_ = LSS.Simulation(np.zeros((dim_x,1)), u)
y = np.array(x)

y = x[0:-1,:]



# Build Hankel Matrix for future f

x=np.array([[1,11],[2,22],[3,33],[4,44],[5,55]])

x_hankel = hankel_matrix(x,f=2)


C = np.array([[1, 0],[0,1]])

A = np.array([[1, 1],[-1,1]])

B = np.array([[1, 1,2],[-1,1,2]])

D = np.array([[1, 1, 1],[1,1,1]])


Of = extend_observ_matrix(C,A,f=2)


Gf = toeplitz(D,(C,B),A,2)

Pu = project_row_space(x_hankel)


solve stuff : Hf Zp (I-Pu)