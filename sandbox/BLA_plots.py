# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:36:19 2022

@author: alexa
"""

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-4,4,200).reshape((-1,1))
y = 0.1*x+np.tanh(x)

plt.plot(x,y)

# Modellansatz x_new = a*x_old

a = (1/(x.T.dot(x))).dot(x.T).dot(y)


y_lin = a*x

plt.plot(x,y_lin)

y_nl = y-y_lin

plt.plot(x,y_nl)