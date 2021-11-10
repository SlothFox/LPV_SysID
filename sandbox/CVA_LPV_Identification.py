# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:07:00 2021

@author: LocalAdmin
"""

from models.NN import LPV_NARX
import numpy as np
from miscellaneous.PreProcessing import arrange_ARX_data
from optim.param_optim import ARXParameterEstimation

import matplotlib.pyplot as plt
# Load data

# 1 identify ARX model (5)
# Arange Data 
# Implement Model equations
# Adapt optimizer?
# 

shifts = 10

NARX = LPV_NARX(dim_u=1,dim_y=1,shifts=shifts,dim_theta=1, initial_params=None, 
                frozen_params = [], init_proc='random')



# Test identification procedure
N=1000
x0 = np.array([[0],[0]])
u = np.random.rand(1000,1).reshape(-1,1)


A = np.array([[0.5,0.5],[0.5, -0.5]])
B = np.array([[0],[0.5]])
C = np.array([[1, 0]])


x = []
y = []

x.append(x0)
y.append(C.dot(x0))

for i in range(1,N):    
    x_new = A.dot(x[i-1]) + B * u[i-1]
    y_new = C.dot(x_new)
    
    x.append(x_new)
    y.append(y_new)


y = np.array(y).reshape(-1,1)

# Arrange NARX Data

# y = np.array([[1 ,11 ],[2, 22],[3, 33],[4, 44],[5, 55]])
# u = np.array([[0.1],[0.2],[0.3],[0.4],[0.5]])

y_ref, y_shift, u_shift = arrange_ARX_data(u=u,y=y,shifts=shifts)

y_ref = y_ref.reshape(1,-1,1)
y_shift = y_shift.reshape(1,-1,shifts)
u_shift = u_shift.reshape(1,-1,shifts)


data = {'u_train':u_shift,'y_train':y_ref, 'y_in':y_shift}


params = ARXParameterEstimation(NARX,data,p_opts=None,s_opts=None, mode='parallel')
print(params)
NARX.Parameters = params


_,y_NARX = NARX.Simulation(y_shift[0,[0],:],u_shift[0])

plt.plot(np.arange(0,1000,1),y)
plt.plot(np.arange(shifts,1000+shifts,1),y_NARX)

