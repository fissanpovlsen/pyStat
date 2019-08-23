#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precalculate covariance matrix given the coordinates and covariance type
Default covariance type: Gaussian with range=1 and sill=1



Created on Sun Apr 30 22:56:34 2017

@author: rasmusmadsen
"""

def precal_cov(xx,yy,covtype='gau',sill=1,rang=1,nugget=0):

    import numpy as np
    
    k = xx.shape[1] # Number of variables in one direction
    n = xx.shape[0] # Number of variables the other direction
    
    num_vars = k*n # Number of variables!

    # Setup positions arrays    
    reshx = xx.reshape(num_vars)
    reshy = yy.reshape(num_vars)
    pos1 = np.array([reshx,reshy]); 
    pos2 = np.array([reshx,reshy]);

    # Preallocate memory for covariance matrix
    cov = np.zeros((num_vars, num_vars), dtype=float)

    # Check if function arguments are correct
    if (covtype != 'gau' and covtype != 'exp' and covtype != 'sph'):
        print('!!!! Unknown covariance type: Try gau, exp or sph !!!!')
        print('!!!! Example: precal_cov(xx,yy,covtype=\'gau\',sill=1,rang=2,nugget=0) which is a !!!!')
        print('!!!! Gaussian covariance type with sill=1, range=2 and nugget = 0 !!!!') 
        return

    for i in range(0, num_vars):
        # Fast vectorized approach
        jj=np.arange(0,num_vars)            
        # positions variable 1
        p1 = np.tile(pos1[:,i], (num_vars, 1)) #Tile: create num_vars by 1 copies of pos1[:,i]
        # positions variable 2
        p2=np.transpose(pos2[:,jj])
        #cov2[i, :] = np.exp(-np.linalg.norm(p1-p2)**2/1**2)
        distances = (p1-p2)**2
        distances = distances.sum(axis=-1)
        distances = np.sqrt(distances)
        if covtype == 'gau':
            #\gamma_z(h)=c_0[1-\exp(-\frac{h^2}{a_0^2})]    
            cov[i, :] = nugget+sill*(np.exp(-distances**2/rang**2))   
        elif covtype == 'exp':
            #\gamma_z(h)=c_0[1-\exp(-\frac{h^2}{a_0^2})]
            cov[i, :] = nugget+sill*(np.exp(-distances/rang))
        elif covtype == 'sph':        
            print('!!!! Spherical covariance type not yet implemented: Try gau or exp !!!!')
    return cov