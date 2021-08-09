# -*- coding: utf-8 -*-
import casadi as cs


def logistic(x):
    
    y = 0.5 + 0.5 * cs.tanh(0.5*x)

    return y

def ReLu(x):
    
    y = np.hstack((np.zeros(x.shape),x))
    
    y = y.max(axis=1).reshape((-1,1))
    
    return y


def RBF(x,c,w):
    d = x-c    
    e = - cs.mtimes(cs.mtimes(d.T,cs.diag(w)**2),d)
    y = cs.exp(e)
    
    return y

def identity(x):
    return x