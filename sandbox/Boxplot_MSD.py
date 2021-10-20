# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:37:02 2021

@author: alexa
"""
# -*- coding: utf-8 -*-
import casadi as cs
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns

from scipy.io import loadmat

import models.NN as NN
from optim import param_optim


path = 'Results/MSD/'
file = 'MSD_RBF_3states.pkl'



res = pkl.load(open(path+'MSD_RBF_3states.pkl','rb'))


res['BFR_test']=res['BFR_test'].astype('float64')


palette = sns.color_palette()[1::]

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})

fig.set_size_inches((9/2.54,4/2.54))

# sns.violinplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette=palette, fliersize=2,ax=axs, linewidth=1)
sns.boxplot(x='dim_theta', y='BFR_test', hue='model',data=res, ax=axs,
               color=".8")
sns.stripplot(x='dim_theta', y='BFR_test', hue='model',data=res, 
                  palette=palette, ax=axs, linewidth=0.1,
                  dodge=True,zorder=1)

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette="Set1",fliersize=2,ax=axs[1], linewidth=1)

axs.legend_.remove()

axs.set_xlabel(r'$\dim(\theta_k)$')

axs.set_ylabel(None)


axs.set_ylim(-5,100)

fig.savefig('Bioreactor_StateSched_Boxplot.png', bbox_inches='tight',dpi=600)

