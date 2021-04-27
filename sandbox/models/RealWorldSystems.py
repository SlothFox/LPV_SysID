# -*- coding: utf-8 -*-
import casadi as cs
import numpy as np

from optim.common import RK4


class RobotManipulator():
    """
    
    """

    def __init__(self,name):
        
        self.name = name
        
        self.Initialize()

    def Initialize(self):
            
            # For convenience of notation
            name = self.name
            
            # Define input, state and output vector
            u = cs.MX.sym('u',2,1)
            x = cs.MX.sym('x',4,1)
            y = cs.MX.sym('y',1,1)
            
            # Sampling time
            dt = 0.1
            
            # Define Model Parameters
            a = cs.MX.sym('a',1,1)
            b = cs.MX.sym('b',1,1)
            c = cs.MX.sym('c',1,1)
            d = cs.MX.sym('d',1,1)
            e = cs.MX.sym('e',1,1)
            f = cs.MX.sym('f',1,1)
            n = cs.MX.sym('n',1,1)
            
            # Put all Parameters in Dictionary with random initialization
            self.Parameters = { 'a': np.array([[5.6794]]),
                                'b': np.array([[1.473]]),
                                'c': np.array([[1.7985]]),
                                'd': np.array([[0.4]]),
                                'e': np.array([[0.4]]),
                                'f': np.array([[2]]),
                                'n': np.array([[1]])}
            
          
            cosd = cs.cos(x[0]-x[1])
            sind = cs.sin(x[0]-x[1])

            M = cs.horzcat(a,b*cosd,b*cosd,c).reshape((2,2)).T
            g = cs.vertcat(d*np.cos(x[0]),
                           e*np.sin(x[1]))
            C = cs.vertcat(b*sind*x[3]**2 + f*x[2],
                           -b*sind*x[2]**2 + f*(x[3]-x[2]))
                           
            
            # continuous dynamics
            x_new = cs.vertcat(x[2::],
                               cs.mtimes(cs.inv(M),n*u-C-g))
            
            input = [x,u,a,b,c,d,e,f,n]
            input_names = ['x','u','a','b','c','d','e','f','n']
            
            output = [x_new]
            output_names = ['x_new']  
            
            
            f_cont = cs.Function(name,input,output,
                                 input_names,output_names)  
            
            x1 = RK4(f_cont,input,dt)
            y1 = x1[0:2]
            
            self.Function = cs.Function(name, input, [x1,y1],
                                        input_names,['x1','y1'])
            
            return None
   
    def OneStepPrediction(self,x0,u0,params=None):
        '''
        Estimates the next state and output from current state and input
        x0: Casadi MX, current state
        u0: Casadi MX, current input
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''
        
        if params==None:
            params = self.Parameters
        
        params_new = []
            
        for name in  self.Function.name_in():
            try:
                params_new.append(params[name])                      # Parameters are already in the right order as expected by Casadi Function
            except:
                continue
        
        x1,y1 = self.Function(x0,u0,*params_new)     
                              
        return x1,y1
   
    def Simulation(self,x0,u,params=None):
        '''
        A iterative application of the OneStepPrediction in order to perform a
        simulation for a whole input trajectory
        x0: Casadi MX, inital state a begin of simulation
        u: Casadi MX,  input trajectory
        params: A dictionary of opti variables, if the parameters of the model
                should be optimized, if None, then the current parameters of
                the model are used
        '''

        x = []
        y = []

        # initial states
        x.append(x0)
                      
        # Simulate Model
        for k in range(u.shape[0]):
            x_new,y_new = self.OneStepPrediction(x[k],u[[k],:],params)
            x.append(x_new)
            y.append(y_new)
        

        # Concatenate list to casadiMX
        y = cs.hcat(y).T    
        x = cs.hcat(x).T
        
        # y = y[0::10]
       
        return y