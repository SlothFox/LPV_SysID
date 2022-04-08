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
file = 'MSD_LPVNN_3states_lam0.01.pkl'
BFR_lin = 71.46                     # BFR linear model on test data


res=pkl.load(open(path+file,'rb'))

# res['dim_theta']=1


# for i in range(len(res)):
#     res.iloc[i,1] = res.iloc[i,1][0]
#     res.iloc[i,0] = res.iloc[i,0][0]


# res['structure'] = None

# res.iloc[0:30]['structure'] = 'A0'
# res.iloc[30:60]['structure'] = 'A1'
# res.iloc[60:90]['structure'] = 'A2'
# res.iloc[90:120]['structure'] = 'A012'
# res.iloc[270:360]['structure'] = 'A3'
# res.iloc[360:450]['structure'] = 'A4'
# res.iloc[450:540]['structure'] = 'A5'
# res.iloc[540:630]['structure'] = 'A6'
# res.iloc[630:720]['structure'] = 'A7'
# res.iloc[720:810]['structure'] = 'A8'

# pkl.dump(res,open(path+file,'wb'))


# Delete NaNs ?

# for i in range(len(res)):
#     try:
#         if np.isnan(res.iloc[i]['BFR_test']) or np.isinf(res.iloc[i]['BFR_test']):
#             # res.drop(res.index[i],inplace = True)
#             continue
#         elif res.iloc[i]['BFR_test']==0.0:
#             res.drop(res.index[i],inplace = True)
#     except:
#         break
     
plt.close('all')
    
palette = sns.color_palette()

# fig, axs = plt.subplots() #plt.subplots(2,gridspec_kw={'height_ratios': [1, 1.5]})

fig, axs = plt.subplots(1,3,gridspec_kw={'width_ratios': [1.5, 1.5, 1.5]})

fig.set_size_inches((9/2.54,4/2.54))
# fig.set_size_inches((9,4))

# sns.boxplot(x='dim_phi', y='BFR_test', hue='structure', data=res, ax=axs,
#             color=".8", linewidth=2)

res = res.loc[res['structure']!='A012']

sns.stripplot(x='structure', y='BFR_test', data=res[res['dim_phi']==1], 
                  palette=palette, ax=axs[0], linewidth=0.3,
                   dodge=True,zorder=1,size=3,marker='x')

sns.stripplot(x='structure', y='BFR_test', data=res[res['dim_phi']==2], 
                  palette=palette, ax=axs[1], linewidth=0.3,
                   dodge=True,zorder=1,size=3,marker='x')
sns.stripplot(x='structure', y='BFR_test', data=res[res['dim_phi']==5], 
                  palette=palette, ax=axs[2], linewidth=0.3,
                   dodge=True,zorder=1,size=3,marker='x')


axs[0].axhline(y=BFR_lin, color='k', linestyle='-',linewidth=1)
axs[1].axhline(y=BFR_lin, color='k', linestyle='-',linewidth=1)
axs[2].axhline(y=BFR_lin, color='k', linestyle='-',linewidth=1)
# sns.stripplot(x='structure', y='BFR_test', hue='dim_phi',data=res, 
#                   palette=palette, ax=axs, linewidth=0.1,
#                    dodge=True,zorder=1,size=5)

# sns.boxplot(x='theta', y='BFR', hue='model',data=BFR_on_val_data, 
#                   palette="Set1",fliersize=2,ax=axs[1], linewidth=1)

# axs.legend_.remove()
# axs.set_xlabel(r'$\dim(\phi_k)$')


axs[0].set_title(r'$\dim(\phi_k)=1$',fontsize=10)
axs[1].set_title(r'$\dim(\phi_k)=2$',fontsize=10)
axs[2].set_title(r'$\dim(\phi_k)=5$',fontsize=10)

axs[0].set_xlabel(None)
axs[1].set_xlabel(None)
axs[2].set_xlabel(None)

axs[0].set_ylabel(None)
axs[1].set_ylabel(None)
axs[2].set_ylabel(None)

xticklabels = [r'$a_{11}$',r'$a_{12}$',r'$a_{13}$',r'$a_{21}$',r'$a_{22}$',
                        r'$a_{23}$',r'$a_{31}$',r'$a_{32}$',r'$a_{33}$']

fontsize = 8

axs[0].set_xticklabels(xticklabels,fontsize=fontsize)
axs[2].set_xticklabels(xticklabels,fontsize=fontsize)
axs[1].set_xticklabels(xticklabels,fontsize=fontsize)

axs[2].set_yticklabels([])
axs[1].set_yticklabels([])


axs[0].set_ylim(55,105)
axs[1].set_ylim(55,105)
axs[2].set_ylim(55,105)
fig.tight_layout()


fig.savefig('MSD_ModelSelection.png', bbox_inches='tight',dpi=600)
