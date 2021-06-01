#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:29:52 2021

@author: alexander
"""

import numpy as np

class PID_Controller():
    
    def __init__(self, P,I,D,AW_limit):
        '''
        

        Parameters
        ----------
        P : float
            P-Channel gain
        I : float
            I-Channel gain
        D : float
            D-Channel gain
        AW_limit: float
            Anti-Wind-Up limit

        Returns
        -------
        None.

        '''

        self.P = P
        self.I = I
        self.D = D
        self.AW_limit = AW_limit
        
        self.e_int = 0
        self.e_diff = 0
        

class LPV_Controller_full():
    
    def __init__(self,Omega = None,vertices = None, x_dim = None, y_dim = None,
                 u_dim = None):
        '''
        Parameters
        ----------
        Omega : list 
            List of parameters of vertex controllers. Each entry is expected
            to be a matrix
                A B 
                C D
        vertices : list
            List of parameters of vertices corresponding to the vertex 
            controllers. Order must be the same as in Omega. Each entry is expected
            to be a tuple
        x_dim : int
            Dimension of the controller state
        y_dim : int
            Dimension of the measured output
        u_dim : int
            Dimension of the control input
        Returns
        -------
        None.

        '''
        
        
        self.Omega = Omega
        self.vertices = vertices
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        
        self.bounds = [(min(v),max(v)) for v in zip(*self.vertices)]


        # The order in which PolytopicCoords_Hypercube() returns the 
        # barycentric coordinates is not defined. In order to find out which
        # coordinate belongs to which vertex, PolytopicCoords_Hypercube() is 
        # called once for each vertex and checked for the '1' entry
        
        self.PolyOrder = np.zeros(len(vertices),dtype=int)
        
        for v in range(0,len(vertices)):
            coords = self.PolytopicCoords_Hypercube(vertices[v])
            self.PolyOrder[v] = int(np.where(coords==1)[0].item())
          

    def CalculateControlInput(self,theta,y,x):
        '''
        Calculates the control input by superposition of the vertex controllers


        Parameters
        ----------
        theta : tuple
            Realization of the affine parameters as the are defined by the 
            LPV model describing the controlled system
        y : array
            Measured system output    
        x : array
            Current controller state    
        
        Returns
        -------
        control input 

        '''
        # Convert affine coordinates to polytopic coordinates first
        alpha = self.PolytopicCoords_Hypercube(theta)        
        
        # Calculate controller state and control input by superposition of all
        # vertex controllers
        Omega_sup = np.zeros((self.x_dim+self.u_dim,self.x_dim+self.y_dim))
        
        for v in range(0,len(self.Omega)):
            Omega_sup =  Omega_sup + alpha[self.PolyOrder[v]]*self.Omega[v]
        
        
        A = Omega_sup[0:self.x_dim,0:self.x_dim]
        B = Omega_sup[0:self.x_dim,self.x_dim::]    
        C = Omega_sup[self.x_dim::,0:self.x_dim]
        D = Omega_sup[self.x_dim::,self.x_dim::]
        
        x_new = np.matmul(A,x) + np.matmul(B,y)
        u_new = np.matmul(C,x_new) + np.matmul(D,y)
        
        
        return x_new,u_new
        
    
    def PolytopicCoords_Hypercube(self, theta):
        
        bounds = self.bounds

        c = np.array(1)        
        
        for i in range(0,len(theta)):
            
            t = (theta[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        
            c = np.vstack((c*(1-t),c*t))
        
        alpha = c
        
        return alpha
