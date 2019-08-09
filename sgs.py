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
import scipy as sp

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
#def cdf( d, bins=12 ):
#    import numpy as np
#    """
#    Function creates a cumulative distribution function, and its inverse, for a given data set d. 
#    The CDF and its inverse are reported as <Nx2> arrays, where N is the number of data points in d, the input data set.
#    """
#    N = len( d )
#    counts, intervals = np.histogram( d, bins=bins )
#    h = np.diff( intervals ) / 2.0
#    f, finv = np.zeros((N,2)), np.zeros((N,2))
#    idx, k, T = 0, 0, float( np.sum( counts ) )
#    for count in counts:
#        for i in range( count ):
#            x = intervals[idx]+h[0]
#            y = np.cumsum( counts[:idx+1] )[-1] / T
#            f[k,:] = x, y 
#            finv[k,:] = d.max()-x,1-y
#            k += 1
#        idx += 1
#    return f, finv

def cdf2(d,num_bins=20):
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
#
#def fit( d ):
#    x, y = d[:,0], d[:,1]
#    def f(t):
#        if t <= x.min():
#            return y[ np.argmin(x) ]
#        elif t >= x.max():
#            return y[ np.argmax(x) ]
#        else:
#            intr = sp.interpolate.interp1d( x, y )
#            return intr(t)
#    return f
#
#
## transform data to normal dist
#def to_norm( data, bins=12 ):
#    mu = np.mean( data )
#    sd = np.std( data )
#    z = ( data - mu ) / sd
#    f, inv = cdf( z, bins=bins )
#    z = sp.stats.norm(0,1).ppf( f[:,1] )
#    z = np.where( z==np.inf, np.nan, z )
#    z = np.where( np.isnan( z ), np.nanmax( z ), z )
#    param = ( mu, sd )
#    return z, inv, param, mu, sd
# 
## transform data from normal dist back
#def from_norm( data, inv, param, mu, sd ):
#    h = fit( inv )
#    f = sp.stats.norm(0,1).cdf( data )
#    z = [ h(i)*sd + mu for i in f ]
#    return z

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
k = 0
remind = np.zeros([50,1])# Remove index
new_dat = np.copy(data)  
for i in range(Nd):
    x_test = np.where((data_sparse[:,0]==data[i,0]))
    y_test = np.where((data_sparse[:,1]==data[i,1]))
    print(i)
    if (x_test[0][0]==y_test[0][0]).all():
        remind[k] = i
        k += 1
    

# Calculate cdfs
num_bins = 10
cdf_data,bin_edges = cdf2(data[:,2],num_bins=num_bins)
cdf_data_sparse,bin_edges_sparse = cdf2(data_sparse[:,2],num_bins=num_bins)
cdf_data_z_sparse,bin_edges_z_sparse = cdf2(data_sparse[:,3],num_bins=num_bins)



#%%

# Figure of testdata
plt.close(1)
plt.figure(1)
plt.subplot(2,3,1)
plt.scatter(data[:,0],data[:,1],c=data[:,2],marker='.')
plt.colorbar()
plt.xlim([-1,20])
plt.ylim([-1,25])
plt.title('Full')

x_cdf= bin_edges[1:]-(bin_edges[1]-bin_edges[0])/2
 
plt.subplot(2,3,4)
plt.plot(x_cdf,cdf_data,label='cdf_normalized')
plt.plot(x_cdf,1-cdf_data,label='cdfinv_normalized')
plt.grid()
plt.title('cdf')
plt.legend()


plt.subplot(2,3,2)
plt.scatter(data_sparse[:,0],data_sparse[:,1],c=data_sparse[:,2],marker='.')
plt.xlim([-1,20])
plt.ylim([-1,25])
plt.colorbar()
plt.title('Sparse')

x_cdf= bin_edges_sparse[1:]-(bin_edges_sparse[1]-bin_edges_sparse[0])/2

plt.subplot(2,3,5)
plt.plot(x_cdf,cdf_data_sparse,label='cdf_normalized')
plt.plot(x_cdf,1-cdf_data_sparse,label='cdfinv_normalized')
plt.grid()
plt.title('cdf')
plt.legend()

plt.subplot(2,3,3)
plt.scatter(data_sparse[:,0],data_sparse[:,1],c=data_sparse[:,3],marker='.')
plt.xlim([-1,20])
plt.ylim([-1,25])
plt.colorbar()
plt.title('Sparse')

x_cdf= bin_edges_z_sparse[1:]-(bin_edges_z_sparse[1]-bin_edges_z_sparse[0])/2

plt.subplot(2,3,6)
plt.plot(x_cdf,cdf_data_sparse,label='cdf_normalized')
plt.plot(x_cdf,1-cdf_data_sparse,label='cdfinv_normalized')
plt.grid()
plt.title('cdf')
plt.legend()
plt.xlim([-3,3])

rfuncs.plot_on_secondary_monitor(1)

get