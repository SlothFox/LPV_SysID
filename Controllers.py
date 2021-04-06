#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:29:52 2021

@author: alexander
"""

import numpy as np

class LPV_Controller_full():
    
    def __init__(self,Omega = None,vertices = None):
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

        Returns
        -------
        None.

        '''
        
        
        self.Omega = Omega
        self.vertices = vertices

        self.bounds = [(min(v),max(v)) for v in zip(*self.vertices)]


        # The order in which PolytopicCoords_Hypercube() returns the 
        # barycentric coordinates is not defined. In order to find out which
        # coordinate belongs to which vertex, PolytopicCoords_Hypercube() is 
        # called once for each vertex and checked for the '1' entry
        
        self.PolytopicCoordsOrder = np.zeros(len(vertices))
        
        for v in range(0,len(vertices)):
            coords = self.PolytopicCoords_Hypercube(vertices[v])
            self.PolytopicCoordsOrder[v] = np.where(coords==1)[0].item()
          

    def CalculateControlInput(self,theta):
        
        
        
        return None
        
    
    def PolytopicCoords_Hypercube(self, theta):
        
        bounds = self.bounds

        c = np.array(1)        
        
        for i in range(0,len(theta)):
            
            t = (theta[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
        
            c = np.vstack((c*(1-t),c*t))
        
        alpha = c
        
        return alpha
