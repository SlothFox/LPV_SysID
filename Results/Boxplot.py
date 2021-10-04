# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:37:02 2021

@author: alexa
"""
# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns

from scipy.io import loadmat

plt.close('all')

path='tanh/log/'

counter = [0,1,2,3,4,5]
# counter = [0,1,2]

for c in counter:

        # Load identification results
        file = path+'Bioreactor_Rehmer_stateSched_2states_'+str(c)+'.pkl'
        ident_results = pkl.load(open(file,'rb'))

        if c==0:
            results = ident_results
        else:
            results = results.append(ident_results,ignore_index=True)
       

# dim theta for lachhab and rehmer is actually dim theta *2 :
# BFR_on_val_data.loc[np.arange(0,50,1),'theta'] = BFR_on_val_data.loc[np.arange(0,50,1),'theta']*2
# BFR_on_val_data.loc[np.arange(100,150,1),'theta'] = BFR_on_val_data.loc[np.arange(100,150,1),'theta']*2

for idx in range(0,60):
    results['BFR_test'][idx] = results['BFR_test'][idx][0]


fig, axs = plt.subplots(1)#plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})
fig.set_size_inches((9/2.54,7/2.54))

palette = sns.color_palette()[1::]

sns.boxplot(x='depth', y='BFR_test', hue='dim_theta',data=results, 
                  palette=palette, fliersize=2,ax=axs, linewidth=1)

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
                  # palette=palette,fliersize=2,ax=axs[1], linewidth=1)

# axs.legend_.remove()
# axs[1].legend_.remove()

# axs.set_xticks([])

axs.set_xlabel('depth')
# axs[0].set_xlabel(None)

# axs[0].set_ylabel(None)
axs.set_ylabel(None)

# axs[1].set_xlim(-0.5,3.5)
# axs[0].set_xlim(-0.5,3.5)

axs.set_ylim(-5,105)
# axs[0].set_ylim(98,100)

fig.savefig('Rehmer_relu.png', bbox_inches='tight',dpi=600)
