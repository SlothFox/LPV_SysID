#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:21:22 2021

@author: alexander
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from miscellaneous.PreProcessing import (hankel_matrix_f,hankel_matrix_p,
                                         extend_observ_matrix, toeplitz, 
                                         project_row_space, hankel_matrix,
                                         oblique_projection)

from models.NN import LinearSSM, LPV_NARX
from optim import param_optim
from sklearn.cross_decomposition import CCA
from scipy.linalg import sqrtm

np.random.seed(1)

N=1000


dim_u = 2
dim_x = 2
dim_y = 2

A = np.array([[0.5, 0.2],[0.7,-0.1]])
B = np.array([[1, 0],[0,1]])
C = np.array([[1, 0],[0,1]])
D = np.array([[0],[0]])

LSS = LinearSSM(dim_u=dim_u,dim_x=dim_x,dim_y=dim_y)

LSS.Parameters = {'A': A,
                  'B': B,
                  'C': C,
                  'D': D}  


u = np.random.randn(N,dim_u)
x,_ = LSS.Simulation(np.zeros((dim_x,1)), u)
y = np.array(x)

y = x[0:-1,:]
y=np.array(y)

IO_data = pd.DataFrame(data=np.hstack((u,y)),columns=['u1','u2','y1','y2'])


# Use data to identify ARX model
ARX = LPV_NARX(dim_u,dim_y,shifts=2, dim_theta=1, initial_params=None, 
                  frozen_params = [], init_proc='random')


order_results = param_optim.ARXOrderSelection(ARX,u,y,order=[i for i in range(1,2)],p_opts=None,
                      s_opts=None)


ARX = LPV_NARX(dim_u,dim_y,shifts=1, dim_theta=1, initial_params=None, 
                 frozen_params = [], init_proc='random')

ARX.Parameters = order_results.loc[0,'params']


''' Use ARX model to remove effects of future inputs with zero initial
conditions'''

# Build matrices with past and future

i=3

z = np.hstack((u,y))

U_hankel = hankel_matrix(u,i=i)
Y_hankel = hankel_matrix(y,i=i)
Z_hankel = hankel_matrix(z,i=i)

Uf= U_hankel[dim_u*i:,:]
Yf= Y_hankel[dim_y*i:,:]

Uf_= U_hankel[dim_u*(i+1):,:]
Yf_= Y_hankel[dim_y*(i+1):,:]

Up= U_hankel[0:dim_u*i,:]
Yp= Y_hankel[0:dim_y*i,:]

Zp = np.vstack((Up,Yp))


# Use ARX model for prediction in each column of Up 

Yf_corr = Yf


y0 = np.zeros((dim_y,1))

for col in range(0,Uf.shape[1]):
    
    uf = Uf[:,[col]].T.reshape((-1,dim_u))
    _,y_pred = ARX.Simulation(y0, uf) 'WERDEN HIER DIE RICHTIGEN ZEITSCHRITTE GEGENÜBERESTELLT? ICH GLAUBE NICHT!!!'
    
    y_pred = y_pred.T.reshape((-1,1))
    
    Yf_corr[:,[col]] = Yf_corr[:,[col]]-y_pred
    

''' Perform CCA between Yf and Zp'''

cca = CCA(n_components=2)

cca.fit(Zp.T, Yf_corr.T)


x_est = (cca.x_loadings_.T[0:2,:]).dot(Zp)
X_c, Y_c = cca.transform(Zp.T, Yf_corr.T)


# OR

W_r = sqrtm(Yf_corr.dot(Yf_corr.T),1/2)
W_c = sqrtm(Zp.dot(Zp.T),1/2)


USV = W_r.dot(Yf_corr.dot(Zp.T)).dot(W_c)

U,S,V = np.linalg.svd(USV,full_matrices=False)

Pf = np.linalg.inv(W_r).dot(U).dot(sqrtm(np.diag(S))) 


''' Estimate Parameters from state space sequence'''

x_est = x_est.T.reshape((1,-1,2))
# x_est = X_c.reshape((1,-1,2))
u_est = u[i:i+N-2*i+1-1,:].reshape((1,-1,2))
y_est = y[i+1:i+1+N-2*i,:].reshape((1,-1,2))


# # Initialize linear SSM and estimate parameters

# # Arrange Training and Validation data in a dictionary with the following
# # structure. The dictionary must have these keys

init_state =np.zeros((1,2,1))

data = {'u_train':u_est, 'y_train':y_est,'init_state_train': init_state,
        'u_val':u_est, 'y_val':y_est,'init_state_val': init_state,
        'u_test':u_est, 'y_test':y_est,'init_state_test': init_state,
        'x_train':x_est}

LSS = LinearSSM(dim_u=dim_u,dim_x=dim_x,dim_y=dim_y)

params_new = param_optim.ModelParameterEstimation(LSS,data, p_opts=None,
                                                    s_opts=None, mode='series')

# # Simulate


LSS.Parameters = params_new

x_sim,y_sim = LSS.Simulation(np.zeros((2,1)), u)

x_sim = np.array(x_sim)
y_sim = np.array(y_sim)

plt.figure()
plt.plot(np.array(y_sim[:,0]))
plt.plot(np.array(y[1:,0]))

plt.figure()
plt.plot(np.array(y_sim[:,1]))
plt.plot(np.array(y[1:,1]))

# Up_= U_hankel[0:dim_u*(i+1),:]
# Yp_= Y_hankel[0:dim_y*(i+1),:]


# Zp = np.vstack((Up,Yp))
# Zp_ = np.vstack((Up_,Yp_))

# O = oblique_projection(Yf,Uf,Zp)
# O_ = oblique_projection(Yf_,Uf_,Zp_)

''' Qin '''
'''
U_f = hankel_matrix_f(u,f=f)[:,p::]
Y_f = hankel_matrix_f(y,f=f)[:,p::]

P_Uf = np.eye(996)-project_row_space(U_f)


U,S,V = np.linalg.svd(Y_f.dot(P_Uf),full_matrices=False)

Gf = U[:,0:2].dot(np.sqrt(np.diag(S[0:2])))

Gf1 = Gf[0:-2,::]
Gf2 = Gf[2::,::]


C_Qin = Gf[0:2,0:2]
A_Qin = np.linalg.inv((Gf1.T).dot(Gf1)).dot(Gf1.T).dot(Gf2)


# Determining B and C is missing
'''

''' Overschee '''

# plt.close('all')
# i=3

# # y = np.array([[11],[12],[13],[14],[15],[16],[17],[18],[19],[20]])
# # u = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
# z = np.hstack((u,y))


# U_hankel = hankel_matrix(u,i=i)
# Y_hankel = hankel_matrix(y,i=i)
# Z_hankel = hankel_matrix(z,i=i)

# Uf= U_hankel[dim_u*i:,:]
# Yf= Y_hankel[dim_y*i:,:]

# Uf_= U_hankel[dim_u*(i+1):,:]
# Yf_= Y_hankel[dim_y*(i+1):,:]

# Up= U_hankel[0:dim_u*i,:]
# Yp= Y_hankel[0:dim_y*i,:]

# Up_= U_hankel[0:dim_u*(i+1),:]
# Yp_= Y_hankel[0:dim_y*(i+1),:]


# Zp = np.vstack((Up,Yp))
# Zp_ = np.vstack((Up_,Yp_))

# O = oblique_projection(Yf,Uf,Zp)
# O_ = oblique_projection(Yf_,Uf_,Zp_)


# W1 = np.eye(6)
# W2 = np.eye(N-2*i+1)-project_row_space(Uf)

# U,S,V = np.linalg.svd(W1.dot(O).dot(W2),full_matrices=False)

# Gf = np.linalg.inv(W1).dot(U[:,0:2]).dot(np.diag(np.sqrt(S[0:2])))
# Gf_ = Gf[0:-dim_u,:]

# Xd = np.linalg.pinv(Gf).dot(O)
# Xd_ = np.linalg.pinv(Gf_).dot(O_)

# a = np.block([[Xd],[U_hankel[i*dim_u:dim_u*(i+1),::]]])
# b = np.block([[Xd_],[Y_hankel[i*dim_y:dim_y*(i+1),::]]])

# ABCD,_,_,_ = np.linalg.lstsq(a.T,b.T)


# # Simulate


# LSS.Parameters = {'A': ABCD[0:2,0:2],
#                   'B': ABCD[0:2,2:4],
#                   'C': ABCD[2:4,0:2],
#                   'D': ABCD[2:4,2:4]}  

# x_sim,y_sim = LSS.Simulation(np.zeros((2,1)), u)

# x_sim = np.array(x_sim)
# y_sim = np.array(y_sim)


# scale_factor = np.array(x_sim)[i,0]/Xd[0,0]

# Xd_int = Xd*scale_factor

# x_new,y_new = LSS.OneStepPrediction(x_sim[3,:], u[3,:])



# plt.figure()
# plt.plot(np.array(y_sim[:,0]))
# plt.plot(np.array(y[:,0]))

# plt.figure()
# plt.plot(np.array(y_sim[:,1]))
# plt.plot(np.array(y[:,1]))

# P_Uf = np.eye(989)-project_row_space(U_f)

# H_fp1 = (Y_f.dot(P_Uf)).dot(Z_p.T)
# H_fp2 = np.linalg.inv((Z_p.dot(P_Uf)).dot(Z_p.T))
# H_fp = H_fp1.dot(H_fp2)

# # H_fp.dot(np.linalg.inv(H_fp))



# Gf = U[:,0:2].dot(np.sqrt(np.diag(S[0:2])))

# Lp = np.linalg.pinv(Gf).dot(H_fp)


# x_est = Xd.T.reshape((1,-1,2))
# u_est = u[3:i+N-2*i+1-1,:].reshape((1,-1,2))
# y_est = y[4:i+1+N-2*i,:].reshape((1,-1,2))


# # Initialize linear SSM and estimate parameters

# # Arrange Training and Validation data in a dictionary with the following
# # structure. The dictionary must have these keys

# init_state =np.zeros((1,2,1))

# data = {'u_train':u_est, 'y_train':y_est,'init_state_train': init_state,
#         'u_val':u_est, 'y_val':y_est,'init_state_val': init_state,
#         'u_test':u_est, 'y_test':y_est,'init_state_test': init_state,
#         'x_train':x_est}

# LSS = LinearSSM(dim_u=dim_u,dim_x=dim_x,dim_y=dim_y)

# params_new = param_optim.ModelParameterEstimation(LSS,data, p_opts=None,
#                                                     s_opts=None, mode='series')



# LSS.Parameters = params_new

# x_sim,y_sim = LSS.Simulation(init_state[0], u)

# plt.figure()
# plt.plot(np.array(y_sim[:,0]))
# plt.plot(np.array(y[1:,0]))

# plt.figure()
# plt.plot(np.array(y_sim[:,1]))
# plt.plot(np.array(y[1:,1]))

# plt.plot(y_sim)
# plt.plot(x_est[0])
# plt.plot(y)

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
