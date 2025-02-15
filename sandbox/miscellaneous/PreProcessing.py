# -*- coding: utf-8 -*-

  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

from numpy.linalg import matrix_power
from scipy.linalg import inv, pinv


def hdf5_to_pd_dataframe(file,save_path=None):
    '''
    

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    # print(file)
    for cycle in file.keys():
        
        try:
            # monitoring charts 1-3
            f3103I = file[cycle]['f3103I_Value']['block0_values'][:,[0,1]]
            f3203I = file[cycle]['f3203I_Value']['block0_values'][:,[0,1]]
            f3303I = file[cycle]['f3303I_Value']['block0_values'][:,[0,1]]    
            timestamp1 = np.vstack([f3103I[:,[0]],f3203I[:,[0]],f3303I[:,[0]]])
               
            # monitoring charts 4-6
            f3403I = file[cycle]['f3403I_Value']['block0_values'][:,[0,1]]
            f3503I = file[cycle]['f3503I_Value']['block0_values'][:,[0,1]]
            f3603I = file[cycle]['f3603I_Value']['block0_values'][:,[0,1]]  
            timestamp2 = np.vstack([f3403I[:,[0]],f3503I[:,[0]],f3603I[:,[0]]])
            
            # measuring chart
            f3113I = file[cycle]['f3113I_Value']['block0_values'][:,0:5]
            f3213I = file[cycle]['f3213I_Value']['block0_values'][:,0:5]
            f3313I = file[cycle]['f3313I_Value']['block0_values'][:,0:5]
            timestamp3 = np.vstack([f3113I[:,[0]],f3213I[:,[0]],f3313I[:,[0]]])       
            
            MonChart1_3 = np.vstack((f3103I,f3203I,f3303I)) 
            MonChart4_6 = np.vstack((f3403I,f3503I,f3603I)) 
            MeasChart = np.vstack((f3113I,f3213I,f3313I)) 
                         
            df1 = pd.DataFrame(data=MonChart1_3[:,[1]],
            index = timestamp1[:,0], columns = ['Q_Vol_ist'])
            
            df2 = pd.DataFrame(data=MonChart4_6[:,[1]],
            index = timestamp2[:,0], columns = ['V_Screw_ist'])
            
            cols = ['p_wkz_ist', 'T_wkz_ist', 'p_inj_soll','p_inj_ist']
    
            df3 = pd.DataFrame(data=MeasChart[:,1:5],
            index = timestamp3[:,0], columns = cols)
            
            df = pd.concat([df1,df2,df3],axis=1)
            
            # now add scalar values
            
            Q_inj_soll = file[cycle]['Q305_Value']['block0_values'][:]
            Q_inj_soll = pd.Series(np.repeat(Q_inj_soll,len(df)))
            df=df.assign(Q_inj_soll = Q_inj_soll.values)
            
            T_zyl1_ist = file[cycle]['T801I_Value']['block0_values'][:]
            T_zyl1_ist = pd.Series(np.repeat(T_zyl1_ist,len(df)))
            df=df.assign(T_zyl1_ist = T_zyl1_ist.values)     
            
            T_zyl2_ist = file[cycle]['T802I_Value']['block0_values'][:]
            T_zyl2_ist = pd.Series(np.repeat(T_zyl2_ist,len(df)))
            df=df.assign(T_zyl2_ist = T_zyl2_ist.values)   
            
            T_zyl3_ist = file[cycle]['T803I_Value']['block0_values'][:]
            T_zyl3_ist = pd.Series(np.repeat(T_zyl3_ist,len(df)))
            df=df.assign(T_zyl3_ist = T_zyl3_ist.values)   
            
            T_zyl4_ist = file[cycle]['T804I_Value']['block0_values'][:]
            T_zyl4_ist = pd.Series(np.repeat(T_zyl4_ist,len(df)))
            df=df.assign(T_zyl4_ist = T_zyl4_ist.values)   
            
            T_zyl5_ist = file[cycle]['T805I_Value']['block0_values'][:]
            T_zyl5_ist = pd.Series(np.repeat(T_zyl5_ist,len(df)))
            df=df.assign(T_zyl5_ist = T_zyl5_ist.values)   
            
            V_um_ist = file[cycle]['V4065_Value']['block0_values'][:]
            df['V_um_ist']=np.nan
            df.loc[0]['V_um_ist'] = V_um_ist
    
            p_um_ist = file[cycle]['p4072_Value']['block0_values'][:]
            df['p_um_ist']=np.nan
            df.loc[0]['p_um_ist'] = p_um_ist

            p_inj_max_ist = file[cycle]['p4055_Value']['block0_values'][:]
            df['p_inj_max_ist']=np.nan
            df.loc[0]['p_inj_max_ist'] = p_inj_max_ist
      
            t_dos_ist = file[cycle]['t4015_Value']['block0_values'][:]
            df['t_dos_ist']=np.nan
            df.loc[0]['t_dos_ist'] = t_dos_ist
            
            t_inj_ist = file[cycle]['t4018_Value']['block0_values'][:]
            df['t_inj_ist']=np.nan
            df.loc[0]['t_inj_ist'] = t_inj_ist

            t_press1_soll = file[cycle]['t312_Value']['block0_values'][:]
            df['t_press1_soll']=np.nan
            df.loc[0]['t_press1_soll'] = t_press1_soll

            t_press2_soll = file[cycle]['t313_Value']['block0_values'][:]
            df['t_press2_soll']=np.nan
            df.loc[0]['t_press2_soll'] = t_press2_soll            
            
            cycle_num = file[cycle]['f071_Value']['block0_values'][0,0]
            df['cycle_num']=np.nan
            df.loc[0]['cycle_num'] = cycle_num
            
            pkl.dump(df,open(save_path+'cycle'+str(cycle_num)+'.pkl','wb'))
        except:
            continue
    
    return None
        

def add_csv_to_pd_dataframe(df_file_path,csv_file_path):
    
    #Read df
    df = pkl.load(open(df_file_path,'rb'))
    
    cycle_num = df.loc[0]['cycle_num']
    
    #Read csv
    df_csv = pd.read_csv(csv_file_path,sep=';',index_col=0)

    # add measurements from csv to pd dataframe
    for key in df_csv.keys():
        df[key]=np.nan
        df.loc[0][key] = df_csv.loc[cycle_num][key]
        
    # some measurements are constant trajectories    
    df['Werkzeugtemperatur'] = df.loc[0]['Werkzeugtemperatur']
    df['Düsentemperatur'] = df.loc[0]['Düsentemperatur']
    df['Einspritzgeschwindigkeit'] = df.loc[0]['Einspritzgeschwindigkeit']
    
    # and need to be renamed    
    df.rename(columns = {'Werkzeugtemperatur':'T_wkz_soll',
                         'Düsentemperatur':'T_nozz_soll',
                         'Einspritzgeschwindigkeit':'v_inj_soll'}, inplace = True)
        
    pkl.dump(df,open(df_file_path,'wb'))
    
    return df
    

def arrange_data_for_qual_ident(cycles,x_lab,q_lab):
    
    x = []
    q = []
    switch = []
    
    for cycle in cycles:
        # Find switching instances
        # Assumption: First switch is were pressure is maximal
        t1 = cycle['p_inj_ist'].idxmax()
        idx_t1 = np.argmin(abs(cycle.index.values-t1))
        
        # Second switch results from 
        t2 = t1 + cycle.loc[0]['t_press1_soll'] + cycle.loc[0]['t_press2_soll']
        
        # find index closest to calculated switching instances
        idx_t2 = np.argmin(abs(cycle.index.values-t2))
        t2 = cycle.index.values[idx_t2]
        
        # Read desired data from dataframe
        temp = cycle[x_lab].values                                              # can contain NaN at the end
        nana = np.isnan(temp).any(axis=1)                                       # find NaN
        
        x.append(temp[~nana,:])
        q.append(cycle.loc[0,q_lab].values)
        switch.append([idx_t1,idx_t2])
   
    # inject = {}
    # press = {}
    # cool = {}
    
    # inject['u'] = df.loc[0:t_um1][u_inj].values[0:-1,:]
    # inject['x'] = df.loc[0:t_um1][x].values[1::,:]
    # inject['x_init'] = df.loc[0][x].values
    
    # press['u'] = df.loc[t_um1:t_um2][u_press].values[0:-1,:]
    # press['x'] = df.loc[t_um1:t_um2][x].values[1::,:]
    # press['x_init'] = df.loc[t_um1][x].values

    # cool['u'] = df.loc[t_um2::][u_cool].values[0:-1,:]
    # cool['x'] = df.loc[t_um2::][x].values[1::,:]
    # cool['x_init'] = df.loc[t_um2][x].values    
    
    return x,q,switch

def arrange_data_for_ident(cycles,y_lab,u_inj_lab,u_press_lab,u_cool_lab,mode):
    
    u = []
    y = []
    switch = []
    
    for cycle in cycles:
        # Find switching instances
        # Assumption: First switch is were pressure is maximal
        t1 = cycle['p_inj_ist'].idxmax()
        idx_t1 = np.argmin(abs(cycle.index.values-t1))
        
        # Second switch 
        t2 = t1 + cycle.loc[0]['t_press1_soll'] + cycle.loc[0]['t_press2_soll']
        idx_t2 = np.argmin(abs(cycle.index.values-t2))
        
        # Third switch, machine opens
        t3 = t2 + cycle.loc[0,'Kühlzeit']
        idx_t3 = np.argmin(abs(cycle.index.values-t3))
        
        # Reduce dataframe to desired variables and treat NaN
        u_lab = u_inj_lab + list(set(u_press_lab) - set(u_inj_lab))
        u_lab = u_lab + list(set(u_cool_lab) - set(u_lab)) 
        
        cycle = cycle[u_lab+y_lab]
        
        if mode == 'quality':
            nan_cycle = np.isnan(cycle[u_lab]).any(axis=1)
            cycle = cycle.loc[~nan_cycle]
            
            y.append(cycle.loc[0,y_lab].values)
            
        elif mode == 'process':
            nan_cycle = np.isnan(cycle).any(axis=1)
            cycle = cycle.loc[~nan_cycle]
            
            y.append(cycle.loc[1:idx_t3+1,y_lab].values)            
        
        # At this point one should test, whether there were NaN not only deleted
        # at the end of the dataframe cycle due to different long charts, 
        # but also if NaN were deleted somewhere "in between"
        # However, I have no time for this right now
        
        # Read desired data from dataframe
        u_inj = cycle[u_inj_lab].values                                         # can contain NaN at the end
        u_press = cycle[u_press_lab].values  
        u_cool = cycle[u_cool_lab].values  


        u_inj = u_inj[0:idx_t1,::]
        u_press = u_press[idx_t1:idx_t2,::]
        u_cool = u_cool[idx_t2:idx_t3,::]
        

        u.append([u_inj,u_press,u_cool])
        switch.append([idx_t1,idx_t2])
   
    return u,y,switch




def eliminate_outliers(doe_plan):
    
    # eliminate all nan
    data = data[data.loc[:,'Gewicht']>=5]
    data = data[data.loc[:,'Stegbreite_Gelenk']>=4]
    data = data[data.loc[:,'Breite_Lasche']>=4]
    
    

# def hankel_matrix(x,f=None,shifts=None):
    
        
#     N = x.shape[0]
#     dim_x = x.shape[1]
    
#     if f is not None:
        
#         if N%f != 0:
#             x = x[0:-1,:]
#             N = x.shape[0] 
        
#         shifts =  int(N/f) + 1
        
#         x_hankel = np.zeros((f,shifts*dim_x))
    
#         for s in range(0,shifts):
#             x_hankel[:,(s)*dim_x:(s+1)*dim_x] = x[s:s+N-shifts+1,:]

#     elif shifts is not None:
#         print('Muss noch implementiert werden!')    
#         x_hankel = None
     

#     return x_hankel

def hankel_matrix_f(x,f=None,shifts=None):
    
        
    N = x.shape[0]
    dim_x = x.shape[1]
    
    if f is not None:
        
        # if N%f != 0:
        #     x = x[0:-1,:]
        #     N = x.shape[0] 
        
        shifts =  N - f + 1
        
        # x = 
        
        x_hankel = np.zeros((dim_x*f,shifts))
    
        for s in range(0,shifts):
            x_hankel[:,[s]] = x[s:s+N-shifts+1,:].reshape((-1,1))

    elif shifts is not None:
        print('Muss noch implementiert werden!')    
        x_hankel = None
     

    return x_hankel

def hankel_matrix_p(x,p=None,shifts=None):
    
        
    N = x.shape[0]
    dim_x = x.shape[1]
    
    if p is not None:
        
        # if N%p != 0:
        #     x = x[0:-1,:]
        #     N = x.shape[0] 
        
        shifts =  N - p + 1
        
        # x = 
        
        x_hankel = np.zeros((dim_x*p,shifts))
    
        for s in range(0,shifts):
            x_p = np.flip(x[N-(s)-p:N-(s),:],axis=0) 
            x_hankel[:,[shifts-(s+1)]] = x_p.reshape((-1,1))

    elif shifts is not None:
        print('Muss noch implementiert werden!')    
        x_hankel = None
     

    return x_hankel


def extend_observ_matrix(C,A,f):
    
    if C.shape[1]==A.shape[0] and A.shape[0]==A.shape[1]:
        dim_x = C.shape[1]
    else:
        ValueError('Dimensionen stimmen nicht!')
    
    dim_y = C.shape[0]
    
    Of = C
    
    for i in range(0,f):
        Of = np.vstack((Of,C.dot(matrix_power(A,f))))
        
    return Of
    
    
def toeplitz(diag,pair,A,f):
    
    D = diag
    C = pair[0]
    B = pair[1]
    
    if C.shape[0] == D.shape[0]:
        dim_y = C.shape[0]
    else:
        ValueError('Dimensionen stimmen nicht!')   
        
    if B.shape[1] == D.shape[1] :
        dim_u = B.shape[1]
    else:
        ValueError('Dimensionen stimmen nicht!')           

    
    Hf = [[np.zeros((dim_y,dim_u)) for row in range(0,f) ] for row in range(0,f)]

    
    for i in range(0,f):
        for j in range(0,f):
  
            if j>i:
                pass
            elif i==j:
                Hf[i][j] = D
            else:
                Hf[i][j] = C.dot(matrix_power(A,f)).dot(B)
    
    Hf = np.block(Hf)
            
    return Hf    
    
def project_row_space(X):

    Px = np.linalg.multi_dot((X.T,pinv(X.dot(X.T)),X))
    
    return Px

def oblique_projection(A,B,C):

    CC = C.dot(C.T)
    CB = C.dot(B.T)
    BB = B.dot(B.T)

    X = np.block([C.T, B.T])
    Y = pinv(np.block([[CC, CB],[CB.T,BB]]))
    
    # r = np.linalg.matrix_rank(C)
    r = C.shape[0]
    
    O = np.linalg.multi_dot((A,X,Y[:,0:r],C))
    
    return O
    
    
def hankel_matrix(x,i=None,shifts=None):
    
        
    N = x.shape[0]
    dim_x = x.shape[1]
    
    if i is not None:

        shifts =  N -2*i + 1
        
        x_hankel = np.zeros((dim_x*(2*i),shifts))
    
        for s in range(0,shifts):
            x_hankel[:,[s]] = x[s:s+N-shifts+1,:].reshape((-1,1))

    elif shifts is not None:
        
        x_hankel = np.zeros((dim_x*(shifts+1),N-shifts))
    
        for s in range(0,x_hankel.shape[1]):
            x_hankel[:,[s]] = x[s:s+shifts+1,:].reshape((-1,1))

    return x_hankel    
    
    
def arrange_ARX_data(u,y=None,shifts=1):
        
        N = y.shape[0]
    
        dim_u = u.shape[1]
        dim_y = y.shape[1]
        
        y_hankel = hankel_matrix(x=y,shifts=shifts)
        u_hankel = hankel_matrix(x=u,shifts=shifts)
        
       
        # y_hankel_rev = np.zeros(y_hankel.shape)
        # u_hankel_rev = np.zeros(u_hankel.shape)
        
        # for block_row in range(0,y_hankel.shape[0]//dim_y):
        #     y_hankel_rev[-(block_row+1)*dim_y:-(block_row)*dim_y,:] = \
        #     y_hankel[block_row*dim_y:(block_row+1)*dim_y,:]

        # for block_row in range(0,u_hankel.shape[0]//dim_u):
        #     u_hankel_rev[-(block_row+1)*dim_u:-(block_row)*dim_u,:] = \
        #     u_hankel[block_row*dim_u:(block_row+1)*dim_u,:]
          
        
        
        y_out = y_hankel[-1*dim_y::,:].T   
        
        y_in = y_hankel[0:-1*dim_y,:]
        u_in = u_hankel[0:-1*dim_u,:]
    
        # reserve order of y_in and u_in block row wise
        y_in = [[y_in[i*dim_y:(i+1)*dim_y,:]] for i in range(0,shifts)]
        y_in.reverse()
        
        u_in = [[u_in[i*dim_u:(i+1)*dim_u,:]] for i in range(0,shifts)]
        u_in.reverse()
        
        y_in = np.block(y_in).T
        u_in = np.block(u_in).T
        
        return y_out, y_in, u_in    
    