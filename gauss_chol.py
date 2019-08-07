#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computing one realization of discrete gaussian 
random field using cholesky decomposition

To compute one realisation of a discrete Gaussian RF:
    1. Compute M independent Gaussian random numbers ξi. 
    2. Compute the Cholesky factorisation of C and 
       store the factor L. 
    3. Compute a matrix-vector product with L.

It takes the mean (mu) and covariance function (C) as inputs
The user can also specify how many realisations is wanted (r) 
By default r=1, if the user does not select anything
                  
Created on Sun Apr 30 00:21:01 2017

@author: rasmusmadsen
"""


def gauss_chol(mu,C,r=1): # r has an equals to, since it is optional and the default value is 1

    # Importing numpy
    import numpy as np
    
    
    # 1. Compute M independent Gaussian random numbers ξi. 
    M = np.shape(C)
    xi = np.random.randn(M[0],r)
    # 2. Compute the Cholesky factorisation of C and 
    #   store the factor L. 
    L = np.linalg.cholesky(C) 
    # 3. Compute a matrix-vector product with L.    
    return mu+np.transpose(L).dot(xi)
    #Z = mu*np.transpose(L)*xi
    
#function Z = gauss_chol(mu,C)
#xi=randn(size(mu));
#R=chol(C);
#Z=mu+R’*xi;