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
Lach = 'MSD_Lachhab_3states_lam0.01.pkl'
RBF = 'MSD_RBF_3states.pkl'
LPVNN = 'MSD_LPVNN_3states_lam0.01.pkl'

Lach=pkl.load(open(path+Lach,'rb'))
RBF=pkl.load(open(path+RBF,'rb'))
LPVNN=pkl.load(open(path+LPVNN,'rb'))


# Pick best from LPVNN
LPVNN = LPVNN.sort_values('BFR_test',ascending=False).iloc[0:10]

# Pick best from Lach
Lach = Lach.sort_values('BFR_test',ascending=False).iloc[0:10]

res = Lach.append([RBF,LPVNN])

for i in range(0,len(res)):
    try:
        res.iloc[i,0] = res.iloc[i,0][0]
        res.iloc[i,1] = res.iloc[i,1][0]
    except:
        continue
    
plt.close('all')
    
palette = sns.color_palette()[1::]

fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})

fig.set_size_inches((9/2.54,5/2.54))

# sns.violinplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette=palette, fliersize=2,ax=axs, linewidth=1)
sns.boxplot(x='dim_theta', y='BFR_test', hue='model',data=res, ax=axs,
               color=".8")
sns.stripplot(x='dim_theta', y='BFR_test', hue='model',data=res, 
                  palette=palette, ax=axs, linewidth=0.1,
                  dodge=True,zorder=1)


axs.set_xlabel(r'$\dim(\theta_k)$',fontsize=10)

axs.set_ylabel(None)

axs.set_yticks([50,75,100])
axs.set_yticklabels([50,75,100])

axs.set_xlim(-0.5,3.5)
axs.set_ylim(30,105)
axs.legend_.remove()

fig.tight_layout()
fig.savefig('MSD_NOE_results.png', bbox_inches='tight',dpi=600)
