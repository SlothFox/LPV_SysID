# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:18:25 2021

@author: LocalAdmin
"""

import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path = 'Results/MSD/'
file = 'MSD_Lachhab_3states_lam0.01.pkl'

BFR_lin = 71.46                     # BFR linear model on test data

res=pkl.load(open(path+file,'rb'))


res['BFR_test']=res['BFR_test'].astype('float64')

     
plt.close('all')
    
palette = sns.color_palette()

fig, axs = plt.subplots()



fig.set_size_inches((9/2.54,4/2.54))

axs.axhline(y=BFR_lin, color='k', linestyle='-',linewidth=1)
sns.stripplot(x='dim_thetaA', y='BFR_test', data=res, 
                  palette=palette, ax=axs, linewidth=0.3,
                   dodge=True,zorder=1,size=3,marker='x')



axs.set_xlabel(r'$\dim(\theta_k)$',fontsize=10)

axs.set_ylabel(None)

axs.set_yticks([60,80,100])
axs.set_yticklabels([60,80,100])


axs.set_ylim(55,105)
fig.tight_layout()


fig.savefig('MSD_ModelSelection_Lachhab.png', bbox_inches='tight',dpi=600)
