# -*- coding: utf-8 -*-
import casadi as cs
from .activations import *


def NN_layer(input,weights,bias,nonlinearity):
    '''
    

    Parameters
    ----------
    input : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.
    bias : TYPE
        DESCRIPTION.
    nonlinearity : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    '''
    
    if nonlinearity == 0:
        nonlin = cs.tanh
    elif nonlinearity == 1:
        nonlin = logistic
    elif nonlinearity == 2:
        nonlin  = identity
    elif nonlinearity == 3:
        nonlin  = ReLu
            
    net = cs.mtimes(weights,input) + bias

    return nonlin(net)