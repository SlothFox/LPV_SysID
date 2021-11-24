#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:25:16 2020

@author: alexander
"""

from sys import path
path.append(r"C:\Users\LocalAdmin\Documents\casadi-windows-py38-v3.5.5-64bit")

import os

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import pickle as pkl

from .DiscreteBoundedPSO import DiscreteBoundedPSO
from .common import OptimValues_to_dict
from .common import BestFitRate,AIC
from models.NN import LinearSSM

from miscellaneous.PreProcessing import arrange_ARX_data

# Import sphere function as objective function
#from pyswarms.utils.functions.single_obj import sphere as f

# Import backend modules
# import pyswarms.backend as P
# from pyswarms.backend.topology import Star
# from pyswarms.discrete.binary import BinaryPSO

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


# from miscellaneous import *


def ControlInput(ref_trajectories,opti_vars,k):
    """
    Übersetzt durch Maschinenparameter parametrierte
    Führungsgrößenverläufe in optimierbare control inputs
    """
    
    control = []
            
    for key in ref_trajectories.keys():
        control.append(ref_trajectories[key](opti_vars,k))
    
    control = cs.vcat(control)

    return control   
    
def CreateOptimVariables(opti, Parameters):
    """
    Defines all parameters, which are part of the optimization problem, as 
    opti variables with appropriate dimensions
    """
    
    # Create empty dictionary
    opti_vars = {}
    
    for param in Parameters.keys():
        dim0 = Parameters[param].shape[0]
        dim1 = Parameters[param].shape[1]
        
        opti_vars[param] = opti.variable(dim0,dim1)
    
    return opti_vars

def ModelTraining(model,data,initializations=10, BFR=False, 
                  p_opts=None, s_opts=None):
    
   
    results = [] 
    
    for i in range(0,initializations):
        
        # initialize model to make sure given initial parameters are assigned
        model.ParameterInitialization()
        
        # Estimate Parameters on training data
        new_params = ModelParameterEstimation(model,data,p_opts,s_opts)
        
        # Assign estimated parameters to model
        for p in new_params.keys():
            model.Parameters[p] = new_params[p]
        
        # Evaluate on Validation data
        u_val = data['u_val']
        y_ref_val = data['y_val']
        init_state_val = data['init_state_val']

        # Evaluate estimated model on validation data        
        e_val = 0
        
        for j in range(0,u_val.shape[0]):   
            # Simulate Model
            pred = model.Simulation(init_state_val[j],u_val[j])
            
            if isinstance(pred, tuple):
                pred = pred[1]
            
            e_val = e_val + cs.sqrt(cs.sumsqr(y_ref_val[j,:,:] - pred))
        
        # Calculate mean error over all validation batches
        e_val = e_val / u_val.shape[0]
        e_val = np.array(e_val).reshape((1,))
        
        
        # Evaluate estimated model on test data
        
        u_test = data['u_test']
        y_ref_test = data['y_test']
        init_state_test = data['init_state_test']
            
        pred = model.Simulation(init_state_test[0],u_test[0])
        
        if isinstance(pred, tuple):
            pred = pred[1]
        
        y_est = np.array(pred)
        
        BFR = BestFitRate(y_ref_test[0],y_est)
        
        # save parameters and performance in list
        results.append([e_val,BFR,model.name,model.dim,i,model.Parameters])
   
    results = pd.DataFrame(data = results, columns = ['loss_val','BFR_test',
                        'model','dim_theta','initialization','params'])
    
    return results 

def HyperParameterPSO(model,data,param_bounds,n_particles,options,
                      initializations=10,p_opts=None,s_opts=None):
    """
    Binary PSO for optimization of Hyper Parameters such as number of layers, 
    number of neurons in hidden layer, dimension of state, etc

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    param_bounds : dict
        A dictionary with structure {'name_of_attribute': [lower_bound,upper_bound]}
    n_particles : int
        Number of particles to use
    options : dict
        options for the PSO, see documentation of toolbox.
    initializations : int, optional
        Number of times the nonlinear optimization problem is solved for 
        each particle. The default is 10.
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    hist, Pandas Dataframe
        Returns Pandas dataframe with the loss associated with each particle 
        in the first column and the corresponding hyperparameters in the 
        second column

    """
    
    
    # Formulate Particle Swarm Optimization Problem
    dimensions_discrete = len(param_bounds.keys())
    lb = []
    ub = []
    
    for param in param_bounds.keys():
        
        lb.append(param_bounds[param][0])
        ub.append(param_bounds[param][1])
    
    bounds= (lb,ub)
    
    # Define PSO Problem
    PSO_problem = DiscreteBoundedPSO(n_particles, dimensions_discrete, 
                                     options, bounds)

    # Make a directory and file for intermediate results 
    os.mkdir(model.name)

    for key in param_bounds.keys():
        param_bounds[key] = np.arange(param_bounds[key][0],
                                      param_bounds[key][1]+1,
                                      dtype = int)
    
    index = pd.MultiIndex.from_product(param_bounds.values(),
                                       names=param_bounds.keys())
    
    hist = pd.DataFrame(index = index, columns=['cost','model_params'])    
    
    pkl.dump(hist, open(model.name +'/' + 'HyperParamPSO_hist.pkl','wb'))
    
    # Define arguments to be passed to vost function
    cost_func_kwargs = {'model': model,
                        'param_bounds': param_bounds,
                        'n_particles': n_particles,
                        'dimensions_discrete': dimensions_discrete,
                        'initializations':initializations,
                        'p_opts': p_opts,
                        's_opts': s_opts}
    
    # Create Cost function
    def PSO_cost_function(swarm_position,**kwargs):
        
        # Load training history to avoid calculating stuff muliple times
        hist = pkl.load(open(model.name +'/' + 'HyperParamPSO_hist.pkl',
                                 'rb'))
            
        # except:
            
        #     os.mkdir(model.name)
            
        #     # If history of training data does not exist, create empty pandas
        #     # dataframe
        #     for key in param_bounds.keys():
        #         param_bounds[key] = np.arange(param_bounds[key][0],
        #                                       param_bounds[key][1]+1,
        #                                       dtype = int)
            
        #     index = pd.MultiIndex.from_product(param_bounds.values(),
        #                                        names=param_bounds.keys())
            
        #     hist = pd.DataFrame(index = index, columns=['cost','model_params'])
        
        # Initialize empty array for costs
        cost = np.zeros((n_particles,1))
    
        for particle in range(0,n_particles):
            
            # Check if results for particle already exist in hist
            idx = tuple(swarm_position[particle].tolist())
            
            if (math.isnan(hist.loc[idx,'cost']) and
            math.isnan(hist.loc[idx,'model_params'])):
                
                # Adjust model parameters according to particle
                for p in range(0,dimensions_discrete):  
                    setattr(model,list(param_bounds.keys())[p],
                            swarm_position[particle,p])
                
                model.Initialize()
                
                # Estimate parameters
                results = ModelTraining(model,data,initializations, 
                                        BFR=False, p_opts=p_opts, 
                                        s_opts=s_opts)
                
                # Save all results of this particle in a file somewhere so that
                # the nonlinear optimization does not have to be done again
                
                pkl.dump(results, open(model.name +'/' + 'particle' + 
                                    str(swarm_position[particle]) + '.pkl',
                                    'wb'))
                
                # calculate best performance over all initializations
                cost[particle] = results.loss.min()
                
                # Save new data to dictionary for future iterations
                hist.loc[idx,'cost'] = cost[particle]
                
                # Save model parameters corresponding to best performance
                idx_min = pd.to_numeric(results['loss'].str[0]).argmin()
                hist.loc[idx,'model_params'] = \
                [results.loc[idx_min,'params']]
                
                # Save DataFrame to File
                pkl.dump(hist, open(model.name +'/' + 'HyperParamPSO_hist.pkl'
                                    ,'wb'))
                
            else:
                cost[particle] = hist.loc[idx].cost.item()
                
        
        
        
        cost = cost.reshape((n_particles,))
        return cost
    
    
    # Solve PSO Optimization Problem
    PSO_problem.optimize(PSO_cost_function, iters=100, **cost_func_kwargs)
    
    # Load intermediate results
    hist = pkl.load(open(model.name +'/' + 'HyperParamPSO_hist.pkl','rb'))
    
    # Delete file with intermediate results
    os.remove(model.name +'/' + 'HyperParamPSO_hist.pkl')
    
    return hist

def ModelParameterEstimation(model,data,p_opts=None,s_opts=None, mode='parallel'):
    """
    

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    values : dict
        dictionary with either the optimal parameters or if the solver did not
        converge the last parameter estimate

    """
    
    
    u = data['u_train']
    y_ref = data['y_train']
    init_state = data['init_state_train']
    
    try:
        x_ref = data['x_train']
    except:
        x_ref = None
        
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create dictionary of all non-frozen parameters to create Opti Variables of 
    OptiParameters = model.Parameters.copy()
    
    for frozen_param in model.FrozenParameters:
        OptiParameters.pop(frozen_param)
        
    
    params_opti = CreateOptimVariables(opti, OptiParameters)
    
    e = 0
    
    ''' Depending on whether a reference trajectory for the hidden state is
    provided or not, the model is either trained in parallel (recurrent) or 
    series-parallel configuration'''
    
    # Training in parallel configuration 
    if mode == 'parallel':
        
        # Loop over all batches 
        for i in range(0,u.shape[0]):   
               
            # Simulate Model
            pred = model.Simulation(init_state[i],u[i],params_opti)
            
            if isinstance(pred, tuple):
                pred = pred[1]
            
            # Calculate simulation error
            e = e + cs.sumsqr(y_ref[i,:,:] - pred)
            
            
    # Training in series parallel configuration        
    elif mode == 'series':
        # Loop over all batches 
        for i in range(0,u.shape[0]):  
            
            # One-Step prediction
            for k in range(u[i,:,:].shape[0]-1):  
                # print(k)
                x_new,y_new = model.OneStepPrediction(x_ref[i,k,:],u[i,k,:],
                                                      params_opti)
            
              
                # Calculate one step prediction error
                e = e + cs.sumsqr(y_ref[i,k,:]-y_new) + \
                    cs.sumsqr(x_ref[i,k+1,:]-x_new) 

    opti.minimize(e)
        
    # Solver options
    if p_opts is None:
        p_opts = {"expand":False}
    if s_opts is None:
        s_opts = {"max_iter": 1000, "print_level":1}

    # Create Solver
    opti.solver("ipopt",p_opts, s_opts)
    
    # Set initial values of Opti Variables as current Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key], model.Parameters[key])
    
    
    # Solve NLP, if solver does not converge, use last solution from opti.debug
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        
    values = OptimValues_to_dict(params_opti,sol)
    
    return values

def EstimateNonlinearStateSequence(model,data,lam):
    """
    

    Parameters
    ----------
    model : LinearSSM
        Linear state space model.
    data : dict
        dictionary containing input and output data as numpy arrays with keys
        'u_train' and 'y_train'
    lam : float
        trade-off parameter between fit to data and linear model fit, needs to
        be positive

    Returns
    -------
    x_LS: array-like
        numpy-array containing the estimated nonlinear state sequence

    """
    
    u = data['u_train'][0]          # [0] because only first batch is used
    y_ref = data['y_train'][0]
    init_state = data['init_state_train'][0]
    
    '''
    Measurement data is assumed to be arranged as
    u0,y1
    u1,y2
    ...
    
    '''
    
    # if not isinstance(model,LinearSSM):
    #     print('Models needs to be',LinearSSM)
              
        # return None
    
    # elif u.shape[0]+1 != y_ref.shape[0]:
    #     print('Length of output signal needs to be lenght of input signal + 1!\n\
    #           [u_0, u_1, ... , u_N] \n\
    #           [y_0, y_1, ... , y_N, y_N+1]')
              
        # return None
    
    # elif lam>=0:
    #     print('lam needs to be greater than zero')
    
    N = u.shape[0]
    num_states = init_state.shape[0]
    

    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    

    # Create decision variables for states
    x_LS = opti.variable(N,num_states) # x0,...,xN-1
    x_LS[0,:] = init_state.T
    
    e = 0
    

    for k in range(0,N-1):
        '''
        Since y0 is not part of recorded data, optimization starts with x1!
        
        x1 = Model(x0,u0)
        y1 = C*x1
        
        e = e + (yref1 - y1)^2 + lam * (xlin2 - x2)^2
        '''

        x1,_ = model.OneStepPrediction(x_LS[[k],:],u[[k],:])
        # print(k)
        y1 = cs.mtimes(model.Parameters['C'],x_LS[[k+1],:].T)
        
        e = e + cs.sumsqr(y_ref[[k],:] - y1)  + \
            + lam * cs.sumsqr(x_LS[[k+1],:].T - x1)   
            
    opti.minimize(e)    
    opti.solver("ipopt")
    
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
            
      
    x_LS = np.array(sol.value(x_LS)).reshape(N,num_states)
    # print(x_LS.shape)
    # x0 was not part of optimization (loop starts at 1!) and is replaced with
    # initial state
    # x_LS[0,:] = init_state.reshape(1,num_states)
    
    return x_LS

def ARXParameterEstimation(model,data,p_opts=None,s_opts=None, mode='parallel'):
    """
    

    Parameters
    ----------
    model : model
        A model whose hyperparameters to be optimized are attributes of this
        object and whose model equations are implemented as a casadi function.
    data : dict
        A dictionary with training and validation data, see ModelTraining()
        for more information
    p_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.
    s_opts : dict, optional
        options to give to the optimizer, see Casadi documentation. The 
        default is None.

    Returns
    -------
    values : dict
        dictionary with either the optimal parameters or if the solver did not
        converge the last parameter estimate

    """
   
    u = data['u_train']
    y_in = data['y_in']
    y_ref = data['y_train']
      
    # Create Instance of the Optimization Problem
    opti = cs.Opti()
    
    # Create dictionary of all non-frozen parameters to create Opti Variables of 
    OptiParameters = model.Parameters.copy()
    
    for frozen_param in model.FrozenParameters:
        OptiParameters.pop(frozen_param)
        
    
    params_opti = CreateOptimVariables(opti, OptiParameters)
    
    e = 0
    
    # Training in series parallel configuration        
    # Loop over all batches 
    for i in range(0,u.shape[0]):  
        
        # One-Step prediction
        for k in range(u[i,:,:].shape[0]-1):  
            # print(k)
            y_new = model.OneStepPrediction(y_in[i,k,:],u[i,k,:],
                                                  params_opti)
        
            # Calculate one step prediction error
            e = e + cs.sumsqr(y_ref[i,k,:]-y_new)

    opti.minimize(e)
        
    # Solver options
    if p_opts is None:
        p_opts = {"expand":False}
    if s_opts is None:
        s_opts = {"max_iter": 3000, "print_level":1}

    # Create Solver
    opti.solver("ipopt",p_opts, s_opts)
    
    # Set initial values of Opti Variables as current Model Parameters
    for key in params_opti:
        opti.set_initial(params_opti[key], model.Parameters[key])
    
    
    # Solve NLP, if solver does not converge, use last solution from opti.debug
    try: 
        sol = opti.solve()
    except:
        sol = opti.debug
        
    values = OptimValues_to_dict(params_opti,sol)
    
    return values

def ARXOrderSelection(model,u,y,order=[i for i in range(1,20)],p_opts=None,
                      s_opts=None):

    results = []
    
    for o in order:
        print(o)
        # Arange data according to model order
       
        y_ref, y_shift, u_shift = arrange_ARX_data(u=u,y=y,shifts=o)

        y_ref = y_ref.reshape(1,-1,1)
        y_shift = y_shift.reshape(1,-1,o)
        u_shift = u_shift.reshape(1,-1,o)


        data = {'u_train':u_shift,'y_train':y_ref, 'y_in':y_shift}
        
        
        setattr(model,'shifts',o)
        model.Initialize()
        
        params = ARXParameterEstimation(model,data)
        
        model.Parameters = params
        
        # Evaluate estimated model on first batch of training data in parallel mode        
        _,y_NARX = model.Simulation(y_shift[0,[0],:],u_shift[0])
        
        y_NARX = np.array(y_NARX)
        
        
        # Calculate AIC
        aic = AIC(y_ref[0],y_NARX,model.num_params,p=2)

        # save results in list
        results.append([o,model.num_params,aic,params])

   
    results = pd.DataFrame(data = results, columns = ['order','num_params',
                                                      'aic','params'])
    
    
    return results
    
    
    
