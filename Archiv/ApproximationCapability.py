# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:56:58 2021

@author: alexa
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

def logistic(x):
    
    y = 0.5 + 0.5 * cs.tanh(0.5*x)

    return y

u = np.linspace(-2,2)
y = -2*u

dim_x = 1
dim_thetaA = 1
fA_dim = 2


# Create Instance of the Optimization Problem
opti = cs.Opti()

x = cs.MX.sym('x',dim_x,1)

A_0 = opti.variable(dim_x,dim_x)
A_lpv = opti.variable(dim_x,dim_thetaA)
W_A = opti.variable(dim_thetaA,dim_x)

W_fA_x = opti.variable(fA_dim,dim_x)
b_fA_h = opti.variable(fA_dim,1)
W_fA = opti.variable(dim_thetaA,fA_dim)
b_fA = opti.variable(dim_thetaA,1)

# Simulate Model
fA_h = logistic(cs.mtimes(W_fA_x,x) + b_fA_h)
fA = logistic(cs.mtimes(W_fA,fA_h)+b_fA)

# fA = 1 

y_est = cs.mtimes(A_0,x) + cs.mtimes(A_lpv, 
    fA*cs.tanh(cs.mtimes(W_A,x))) 

theta = fA * cs.tanh(cs.mtimes(W_A,x))/cs.mtimes(W_A,x)


f = cs.Function('f',[x,A_0,A_lpv,W_A,W_fA_x,b_fA_h,W_fA,b_fA],[y_est])


f_theta = cs.Function('f_theta',[x,A_0,A_lpv,W_A,W_fA_x,b_fA_h,W_fA,b_fA],[theta])

  
e = 0

# Loop over all experiments
for i in range(0,u.shape[0]):
     
    # e = e + cs.sumsqr(y[i] - f(u[i],A_0,A_lpv,W_A,W_fA_x,b_fA_h,W_fA,b_fA))
    e = e + cs.sumsqr(y[i] - f(u[i],A_0,A_lpv,W_A,W_fA_x,b_fA_h,W_fA,b_fA))
    
opti.minimize(e)
    
# Create Solver
opti.solver("ipopt")

opti.set_initial(A_0,np.random.normal(0,0.1,(dim_x,dim_x)))
opti.set_initial(A_lpv,np.random.normal(0,3,(dim_x,dim_thetaA)))
opti.set_initial(W_A,np.random.normal(0,3,(dim_thetaA,dim_x)))
opti.set_initial(W_fA_x,np.random.normal(0,3,(fA_dim,dim_x)))
opti.set_initial(b_fA_h,np.random.normal(0,3,(fA_dim,1)))
opti.set_initial(W_fA,np.random.normal(0,3,(dim_thetaA,fA_dim)))
opti.set_initial(b_fA,np.random.normal(0,3,(dim_thetaA,1)))

try:
    sol = opti.solve()
    
    A_0 = sol.value(A_0)
    A_lpv = sol.value(A_lpv)
    W_A = sol.value(W_A)
    W_fA_x = sol.value(W_fA_x)
    b_fA_h = sol.value(b_fA_h)
    W_fA = sol.value(W_fA)
    b_fA = sol.value(b_fA)

except:
    A_0 = opti.debug.value(A_0)
    A_lpv = opti.debug.value(A_lpv)
    W_A = opti.debug.value(W_A)
    W_fA_x = opti.debug.value(W_fA_x)
    b_fA_h = opti.debug.value(b_fA_h)
    W_fA = opti.debug.value(W_fA)
    b_fA = opti.debug.value(b_fA)

y_est = []
theta_est = []

# u_test = np.linspace(-5,5,1000)


for i in range(0,u.shape[0]):

    y_est.append(f(u[i],A_0,A_lpv,W_A,W_fA_x,b_fA_h,W_fA,b_fA))
    theta_est.append(f_theta(u[i],A_0,A_lpv,W_A,W_fA_x,b_fA_h,W_fA,b_fA))

y_est = np.array(y_est)
theta_est = np.array(theta_est)


plt.figure()
plt.plot(u,y)
plt.plot(u,y_est)
plt.plot(u,theta_est)

# plt.plot(u,theta_est)


# plt.close('all')


