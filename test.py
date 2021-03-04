# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:08:40 2021

@author: alexa
"""
def RBF(x,c,w):
    d = x-c    
    e = - cs.mtimes(cs.mtimes(d.T,cs.diag(w)**2),d)
    y = cs.exp(e)
    
    return y




A = np.array([[0.7,-0.1],[0.3,0.3]])
B = np.array([[1,0],[0,-2]])
C = np.array([[1,0],[0,1]])
O = np.array([[0],[0]])
c_u0 = np.array([[0],[0]])
c_x0 = np.array([[0],[0]])
w_u0 = np.array([[1],[1]])
w_x0 = np.array([[1],[1]])


x0 = init_state[0]
u0 = u[0,0,:].reshape((2,1))



# Define Model Equations, loop over all local models
x_new = 0
r_sum = 0


c = cs.vertcat(c_x0,c_u0)
w = cs.vertcat(w_x0,w_u0)

r = RBF(cs.vertcat(x,u),c,w)

x_new = x_new + \
r * (cs.mtimes(A,x) + cs.mtimes(B,u) + O)

r_sum = r_sum + r

x_new = x_new / (r_sum + 1e-06)

y_new = 0
r_sum = 0

c = cs.vertcat(c_x0,c_u0)
w = cs.vertcat(w_x0,w_u0)
    
r = RBF(cs.vertcat(x_new,u),c,w)
    
y_new = y_new + r * (cs.mtimes(C,x_new))
    
r_sum = r_sum + r
    
y_new = y_new / (r_sum + 1e-06)