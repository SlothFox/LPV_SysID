#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:21:22 2021

@author: alexander
"""
import numpy as np 
import matplotlib.pyplot as plt

from miscellaneous.PreProcessing import (hankel_matrix_f,hankel_matrix_p,
                                         extend_observ_matrix, toeplitz, project_row_space)

from models.NN import LinearSSM
from optim import param_optim

N=1000

p=2
f=10

dim_u = 2
dim_x = 2
dim_y = 2

A = np.array([[0.5, 0.6],[0.5,0.3]])
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
y=np.array(y)


z = np.hstack((u,y))

Z_p = hankel_matrix_p(z,p=p)[:,0:-f]
U_f = hankel_matrix_f(u,f=f)[:,p::]
Y_f = hankel_matrix_f(y,f=f)[:,p::]
P_Uf = np.eye(989)-project_row_space(U_f)

H_fp1 = (Y_f.dot(P_Uf)).dot(Z_p.T)
H_fp2 = np.linalg.inv((Z_p.dot(P_Uf)).dot(Z_p.T))
H_fp = H_fp1.dot(H_fp2)

# H_fp.dot(np.linalg.inv(H_fp))

U,S,V = np.linalg.svd(H_fp.dot(Z_p),full_matrices=False)

Gf = U[:,0:2].dot(np.sqrt(np.diag(S[0:2])))

Lp = np.linalg.pinv(Gf).dot(H_fp)


x_est = Lp.dot(Z_p).reshape((1,-1,2))
u_est = u[p:-f+1,:].reshape((1,-1,2))
y_est = y[p:-f+1,:].reshape((1,-1,2))

# Initialize linear SSM and estimate parameters

# Arrange Training and Validation data in a dictionary with the following
# structure. The dictionary must have these keys

init_state =np.zeros((1,2,1))

data = {'u_train':u_est, 'y_train':y_est,'init_state_train': init_state,
        'u_val':u_est, 'y_val':y_est,'init_state_val': init_state,
        'u_test':u_est, 'y_test':y_est,'init_state_test': init_state,
        'x_train':x_est}

LSS.Parameters['C'] = np.array([[1, 1],[1,1]])

params_new = param_optim.ModelParameterEstimation(LSS,data, p_opts=None,
                                                   s_opts=None, mode='series')



LSS.Parameters = params_new

x_sim,y_sim = LSS.Simulation(init_state[0], u)

x_sim = np.array(x_sim)
y_sim = np.array(y_sim)

plt.plot(y_sim)
plt.plot(x_est[0])
plt.plot(y)

# Build Hankel Matrix for future f

# x=np.array([[1,11],[2,22],[3,33],[4,44],[5,55]])

# z = np.array([[1,11],[2,22],[3,33],[4,44],[5,55],[6,66],[7,77],[8,88],[9,99],[10,1010]])
# Z_p = hankel_matrix_p(z,p=3)[:,0:-5]


# f=3
# p=3

# y = np.array([[11],[12],[13],[14],[15],[16],[17],[18],[19],[20]])
# u = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
# z = np.hstack((u,y))


# U_f = hankel_matrix_f(u,f=f)[:,p::]
# Y_f = hankel_matrix_f(y,f=f)[:,p::]
# Z_p = hankel_matrix_p(z,p=p)[:,0:-f]
# H_f = toeplitz(D,(C,B),A,2)
# G_f = 

# extend_observ_matrix(C,A,f=2)
# G_f




# Of = 


# Gf = 

# Pu = project_row_space(x_hankel)


# solve stuff : Hf Zp (I-Pu)