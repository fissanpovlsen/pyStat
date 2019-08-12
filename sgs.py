# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:14:42 2019

@author: Rasmus Bødker Madsen

To perform Sequential Gaussian Simulation of sparse data

Algortihm: (from http://connor-johnson.com/2014/06/29/two-dimensional-sequential-gaussian-simulation-in-python/)
1. Define a grid, this may be fine or coarse
2. Place the z-transformed initial data into the nearest grid cells
3. Visit the rest of the grid cells at random and perform kriging using all of the data available
4. Back-transform the data to retrieve the approximate distribution

There are two important things to note here:

- SGS expects normally distributed data, so if the data is not normally distributed, 
  we use a z-score transformation before SGS and a back-transformation afterwards
- In step 3. above, as we move randomly about the grid, we add each newly kriged estimate
  to the data used for kriging each time

The second point means that our data set for kriging increases by one data point for each step until we visit all of the cells in our grid. The randomization allows us to run multiple simulations and then take their mean.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import rasmus_py_funcs as rfuncs
import scipy.stats as stats
import scipy as sp
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

#import time
#import scipy.stats
#import scipy.optimize
#import scipy.interpolate
# 
#import krige
#import utilities
# 
#import random
#
def SVh( P, h, bw ):
    from scipy.spatial.distance import pdist, squareform
    '''
    Experimental semivariogram for a single lag
    '''
    pd = squareform( pdist( P[:,:2] ) )
    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )
    return np.sum( Z ) / ( 2.0 * len( Z ) )
 
def SV( P, hs, bw ):
    '''
    Experimental variogram for a collection of lags
    '''
    sv = list()
    for h in hs:
        sv.append( SVh( P, h, bw ) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]
    return np.array( sv ).T
 
def C( P, h, bw ):
    '''
    Calculate the sill
    '''
    c0 = np.var( P[:,2] )
    if h == 0:
        return c0
    return c0 - SVh( P, h, bw )




def calculate_cdf(d,num_bins=20):
    import numpy as np
    counts, bin_edges = np.histogram (d, bins=num_bins)
    cdf = np.cumsum (counts)
    cdf = cdf/cdf[-1]
    #plt.plot (bin_edges[1:], cdf/cdf[-1])
    #N = np.size(d)
    #cdf = np.zeros((N,2))
    #cdf[:,0] = np.cumsum(d)/np.sum(d)
    #cdf[:,1] = 1-cdf[:,0]    
    return cdf,bin_edges

def get_index_for_data_on_grid(data_sparse,data):
    k = 0
    duplicate = np.zeros([50,1])# Duplicate index
    for i in range(Nd):
        x_test = np.where((data_sparse[:,0]==data[i,0]))
        y_test = np.where((data_sparse[:,1]==data[i,1]))
        if (len(x_test[0])>0 and len(y_test[0])>0):
            for ii in range(len(x_test[0])):
                for iii in range(len(y_test[0])):
                    if (x_test[0][ii]==y_test[0][iii]).any():
                        duplicate[k] = i
                        k += 1
                        print(k)
    non_duplicate = np.delete(np.arange(500),[duplicate]) # Non-duplicate index
    return non_duplicate,duplicate
    
def opt( fct, x, y, C0, parameterRange=None, meshSize=1000 ):
    if parameterRange == None:
        parameterRange = [ x[1], x[-1] ]
    mse = np.zeros( meshSize )
    a = np.linspace( parameterRange[0], parameterRange[1], meshSize )
    for i in range( meshSize ):
        mse[i] = np.mean( ( y - fct( x, a[i], C0 ) )**2.0 )
    return a[ mse.argmin() ]

def spherical( h, a, C0 ):
    '''
    Spherical model of the semivariogram
    '''
    # if h is a single digit
    if type(h) == np.float64:
        # calculate the spherical function
        if h <= a:
            return C0*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    # if h is an iterable
    else:
        # calcualte the spherical function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        return map( spherical, h, a, C0 )


def cvmodel( P, model, hs, bw ):
    '''
    Input:  (P)      ndarray, data
            (model)  modeling function
                      - spherical
                      - exponential
                      - gaussian
            (hs)     distances
            (bw)     bandwidth
    Output: (covfct) function modeling the covariance
    '''
    # calculate the semivariogram
    sv = SV( P, hs, bw )
    # calculate the sill
    C0 = C( P, hs[0], bw )
    # calculate the optimal parameters
    param = opt( model, sv[0], sv[1], C0 )
    # return a covariance function
    covfct = lambda h, a=param: C0 - model( h, a, C0 )
    return covfct

# Create some data for testing
Nd = 500    

d = np.ones(Nd)
d[0:100] = np.random.rand(100)*1
d[100:200] = np.random.rand(100)*2
d[200:300] = np.random.rand(100)*3
d[300:400] = np.random.rand(100)*2.5
d[400:500] = np.random.rand(100)*1.5


#### 1. Define a grid
x = np.arange(0,20)
y = np.arange(0,25)


# Initialize data structure
data = np.zeros([Nd,3])
data_sparse = np.zeros([50,4])

# Fill in data structure
data[:,0] = np.repeat(x,25,axis=0)
data[:,1] = np.tile(y,20)
data[:,2] = np.copy(d)



# Take out 50 random points for sparse 
data_sparse[:,0:3] = data[np.random.choice(Nd,50, replace=False),:]

# 2. Place the z-transformed initial data into the nearest grid cells
data_sparse[:,3] = sp.stats.zscore(data_sparse[:,2]) # Calculate z-score


# 3. Visit the rest of the grid cells at random and perform kriging using all of the data available
# Define random path
non_duplicate,duplicate = get_index_for_data_on_grid(data_sparse,data) # Getting indexes where there is no data
randpath = np.random.permutation(non_duplicate) # randomize this array



# Calculate cdfs
num_bins = 10
cdf_data,bin_edges = calculate_cdf(data[:,2],num_bins=num_bins)
cdf_data_sparse,bin_edges_sparse = calculate_cdf(data_sparse[:,2],num_bins=num_bins)
cdf_data_z_sparse,bin_edges_z_sparse = calculate_cdf(data_sparse[:,3],num_bins=num_bins)


plt.close(2)
plt.figure(2)

sp = cvmodel( data, model=spherical, hs=np.arange(0,50,5), bw=5 )
plt.plot( sv[0], sv[1], '.-' )
plt.plot( sv[0], sp( sv[0] ) ) ;
plt.title('Spherical Model')
plt.ylabel('Semivariance')
plt.xlabel('Lag [m]')
#savefig('semivariogram_model.png',fmt='png',dpi=200)

#%%

# Figure of testdata
plt.close(1)
plt.figure(1)
plt.subplot(3,3,1)
plt.scatter(data[:,0],data[:,1],c=data[:,2],marker='.')
plt.colorbar()
plt.xlim([-1,20])
plt.ylim([-1,25])
plt.title('Full')

x_cdf= bin_edges[1:]-(bin_edges[1]-bin_edges[0])/2
 
plt.subplot(3,3,4)
plt.plot(x_cdf,cdf_data,label='cdf_normalized')
plt.plot(x_cdf,1-cdf_data,label='cdfinv_normalized')
plt.grid()
plt.title('cdf')
plt.legend()


plt.subplot(3,3,7)
# bandwidth, plus or minus 250 meters
bw = 5
# lags in 500 meter increments from zero to 10,000
hs = np.arange(0,50,bw)
sv = SV( data, hs, bw )
plt.plot( sv[0], 1-sv[1], '.-' )
plt.xlabel('Lag [m]')
plt.ylabel('Semivariance')
plt.title('Sample Semivariogram') ;


plt.subplot(3,3,2)
plt.scatter(data_sparse[:,0],data_sparse[:,1],c=data_sparse[:,2],marker='.')
plt.xlim([-1,20])
plt.ylim([-1,25])
plt.colorbar()
plt.title('Sparse')

x_cdf= bin_edges_sparse[1:]-(bin_edges_sparse[1]-bin_edges_sparse[0])/2

plt.subplot(3,3,5)
plt.plot(x_cdf,cdf_data_sparse,label='cdf_normalized')
plt.plot(x_cdf,1-cdf_data_sparse,label='cdfinv_normalized')
plt.grid()
plt.title('cdf')
plt.legend()


plt.subplot(3,3,8)
sv_sparse = SV( data_sparse, hs, bw )
plt.plot( sv_sparse[0], 1-sv_sparse[1], '.-' )
plt.xlabel('Lag [m]')
plt.ylabel('Semivariance')
plt.title('Sample Semivariogram') ;


plt.subplot(3,3,3)
plt.scatter(data_sparse[:,0],data_sparse[:,1],c=data_sparse[:,3],marker='.')
plt.xlim([-1,20])
plt.ylim([-1,25])
plt.colorbar()
plt.title('Sparse')

x_cdf= bin_edges_z_sparse[1:]-(bin_edges_z_sparse[1]-bin_edges_z_sparse[0])/2

plt.subplot(3,3,6)
plt.plot(x_cdf,cdf_data_sparse,label='cdf_normalized')
plt.plot(x_cdf,1-cdf_data_sparse,label='cdfinv_normalized')
plt.grid()
plt.title('cdf')
plt.legend()
plt.xlim([-3,3])

rfuncs.plot_on_secondary_monitor(1)

