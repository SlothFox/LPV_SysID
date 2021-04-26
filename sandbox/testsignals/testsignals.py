# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt



def APRBS(N,step_range,holding_range):
    '''Signal generator for Amplitude Modulated Pseudo Random Binary Sequence.

    Parameters
    ----------
        step_range :     array-like
                         Array, list or tuple with two entries. First entry defines
                         lower bound of the admitted signal range, second value
                         defines upper bound of admitted signal range.
        holding_range:   array-like
                         Array, list or tuple with two entries. First entry defines
                         lower bound on the holding time, second value
                         defines upper bound on the holding time.
                         
    Returns:
        APRBS: numpy array of dimension (1,N)

    '''
    
    # random signal generation
    
    
    steps = np.random.rand(N) * (step_range[1]-step_range[0]) + step_range[0] # range for amplitude
    
    holding_time = np.random.rand(N) *(holding_range[1]-holding_range[0]) + holding_range[0] # range for frequency
    holding_time = np.round(holding_time)
    holding_time = holding_time.astype(int)
    
    holding_time[0] = 0
    
    for i in range(1,np.size(holding_time)):
        holding_time[i] = holding_time[i-1]+holding_time[i]
    
    # Random Signal
    i=0
    APRBS = np.zeros((1,N))
    while holding_time[i]<np.size(APRBS):
        k = holding_time[i]
        APRBS[k:] = steps[i]
        i=i+1
    
    return APRBS
