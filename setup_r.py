#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
" Henning Omre NTNU
" Variance estimation

Created on Sat Apr 29 23:10:16 2017

@author: rasmusmadsen
"""
"""
%% Setup r (Gaussian distribution)
% r = N(mu_r*I_nk,sigma_r^2*sigma_r_0)


%      <---------------k---------------> 
%   ^   |   |   |   |   |   |   |   |   |
%   |   |   |   |   |   |   |   |   |   |
%   |   (   (   (   (   (   (   (   (   (
%   |   |   |   |   |   |   |   |   |   |   
%   n   |   |   |   |   |   |   |   |   |
%   |   (   (   (   (   (   (   (   (   (
%   |   )   )   )   )   )   )   )   )   )
%   |   )   )   )   )   )   )   )   )   )
%   v   |   |   |   |   |   |   |   |   |

 k: number of traces
 n: number of samples per trace
"""

import numpy as np
import matplotlib.pyplot as plt
#from tictoc import *
from gauss_chol import gauss_chol
from precal_cov import precal_cov

# x-location (TRACES)
k = 31 # number of traces
x = np.arange(1,k+1) #traces

# y-location (SAMPLES)
n = 91 # number of samples per trace
y = np.arange(1,n+1) #traces

# Meshgrid
xx, yy = np.meshgrid(x, y)

# Setup expected value of r
mu_r = 1
# Identity matrix of size (n*k)x1
I_nk = np.ones((n*k,1)); #Identity with size (n*k)x1

# Setup r covarince matrix
sigma_r_2 = 0.001  # Variance (scalar)
#V = ?sill Sph(range,rotation,anisotropy_factor)?
#sigma_r_0=precal_cov([xx(:) yy(:)],[xx(:) yy(:)],'1 Sph(5,0,5)'); % size (n*k)x(n*k)
sigma_r_0 = sigma_r_2*mu_r*np.eye(k*n,k*n); # Independent

# Calculate covariance matrix
cov = precal_cov(xx,yy,'exp',1,4)

# initiate Figure 1 and clear it
fig1 = plt.figure(1); plt.clf()

# Plot covariance matrix
plt.subplot(121)
plt.imshow(cov) # Show realization of cov as colorimage
plt.colorbar() # colorbar
plt.title('cov') # title
                     
# Seeding random generator for realization below
np.random.seed(1)# Seeding with 1
#np.random.seed() # Seeding according to time and date

# Generating a realization of r using cholesky decomposition
r_vec = gauss_chol(mu_r*I_nk,sigma_r_2*cov,1)

# Reshaping r_vec
r = np.reshape(r_vec,(n,k))

# Plot realization of r    
plt.subplot(122)        
plt.imshow(r) # Show realization of r as colorimage
plt.colorbar() # colorbar
plt.xlabel('Traces') # Xlabel
plt.ylabel('Samples') # Ylabel
plt.title('r') # title
fig1.canvas.manager.window.raise_() # Raise plot to the front
plt.show()                                 
                                   
                                 
                                 
                                 
