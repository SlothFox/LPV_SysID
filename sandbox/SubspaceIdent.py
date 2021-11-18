#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:21:22 2021

@author: alexander
"""
import numpy as np 


from miscellaneous.PreProcessing import (hankel_matrix_f,hankel_matrix_p,
                                         extend_observ_matrix, toeplitz, project_row_space)

from models.NN import LinearSSM

N=1000

dim_u = 2
dim_x = 2
dim_y = 2

A = np.array([[0.9, 0.5],[0.5,0.8]])
B = np.array([[1, 0],[0,1]])
C = np.array([[1, 0],[0,1]])
D = np.array([[0],[0]])

LSS = LinearSSM(dim_u=dim_u,dim_x=dim_x,dim_y=dim_y)

LSS.Parameters = {'A': A,
                  'B': B,
                  'C': C}  


u = np.random.randn(N,dim_u)
x,_ = LSS.Simulation(np.zeros((dim_x,1)), u)
y = np.array(x)

y = x[0:-1,:]

z = np.hstack((u,y))

z = np.array([[1,11],[2,22],[3,33],[4,44],[5,55],[6,66],[7,77]])


# Build Hankel Matrix for future f

# x=np.array([[1,11],[2,22],[3,33],[4,44],[5,55]])

Z_p = hankel_matrix_p(z,p=2)

# U_f = hankel_matrix_f(u,f=2)
# y_f = hankel_matrix_f(y,f=2)

# H_f = toeplitz(D,(C,B),A,2)
# G_f = 

# extend_observ_matrix(C,A,f=2)
# G_f




# Of = 


# Gf = 

# Pu = project_row_space(x_hankel)


# solve stuff : Hf Zp (I-Pu)