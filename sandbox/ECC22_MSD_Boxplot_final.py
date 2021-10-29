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
Lach = 'MSD_Lachhab_3states_NOE2_lam0.01.pkl'
RBF = 'MSD_RBF_3states.pkl'
LPVNN_NOE = 'LPVNN_NOE_final.pkl'
LPVNN_init = 'MSD_LPVNN_3states_lam0.01.pkl'


BFR_lin = 71.46                     # BFR linear model on test dat

Lach=pkl.load(open(path+Lach,'rb'))
RBF=pkl.load(open(path+RBF,'rb'))
LPVNN_NOE=pkl.load(open(path+LPVNN_NOE,'rb'))
LPVNN_init=pkl.load(open(path+LPVNN_init,'rb'))


# for i in range(22,30):
#     df = pkl.load(open(path+'LPVNN_NOE_'+str(i),'rb'))
    
#     try:
#         LPVNN_NOE_final = LPVNN_NOE_final.append(df)
#     except:
#         LPVNN_NOE_final = df
        
# pkl.dump(LPVNN_NOE_final,open(path+'LPVNN_NOE_final.pkl','wb'))


# Pick best from LPVNN
LPVNN_NOE = LPVNN_NOE.sort_values('BFR_test',ascending=False).iloc[0:2]
LPVNN_init = LPVNN_init.sort_values('BFR_test',ascending=False).iloc[0:9]

LPVNN_NOE = LPVNN_NOE.append(LPVNN_init)



# Pick best from Lach
Lach = Lach.sort_values('BFR_test',ascending=False).iloc[8:19]
Lach['dim_theta']=1

res = Lach.append([RBF,LPVNN_NOE])

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

axs.axhline(y=BFR_lin, color='k', linestyle='-',linewidth=1)

axs.set_xlabel(r'$\dim(\theta_k)$',fontsize=10)

axs.set_ylabel(None)

axs.set_yticks([50,75,100])
axs.set_yticklabels([50,75,100])

axs.set_xlim(-0.5,3.5)
axs.set_ylim(30,105)
axs.legend_.remove()

fig.tight_layout()
fig.savefig('MSD_NOE_results.png', bbox_inches='tight',dpi=600)
