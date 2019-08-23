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
import matplotlib.pyplot as plt
import numpy as np
import GEUS_RESPROB.rasmus_py_funcs as rfuncs
import scipy.stats as stats
import scipy as sp
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

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

def get_index_for_data_on_grid(data_sparse,xx,yy):
    k = 0
    grid_x = xx.flatten()
    grid_y = yy.flatten()
    Nd = np.size(grid_x,0)
    Nd_sparse = np.size(data_sparse,0)
    duplicate = np.zeros([Nd_sparse,1])# Duplicate index
    for i in range(Nd):
        x_test = np.where((data_sparse[:,0]==grid_x[i]))
        y_test = np.where((data_sparse[:,1]==grid_y[i]))
        if (len(x_test[0])>0 and len(y_test[0])>0):
            for ii in range(len(x_test[0])):
                for iii in range(len(y_test[0])):
                    if (x_test[0][ii]==y_test[0][iii]).any():
                        duplicate[k] = i
                        k += 1
    duplicate = duplicate.astype(int)
    non_duplicate = np.delete(np.arange(Nd),[duplicate]) # Non-duplicate index
    return non_duplicate,duplicate

def get_index_to_nearest_neighbours(data_x,data_y,pt,k=10):
    import scipy as sp
    import numpy as np

    data_coords = np.array([data_x,data_y]).transpose()
    hej = sp.spatial.distance.cdist(data_coords,pt,'euclidean')
    indexlist = np.argsort(hej,axis=0)[0:k] # Sort indexes of list 
    return indexlist # Take k nearest neigbours

# 3. Visit the rest of the grid cells at random and perform kriging using all of the data available
def sgs(data_x,data_y,data_val,xx,yy,seed=1,k=10):
    
    kriging_points = np.array([data_x,data_y,data_val]).transpose()

    grid_x = xx.flatten()
    grid_y = yy.flatten()
    
    non_duplicate,duplicate = get_index_for_data_on_grid(kriging_points[:,0:2],grid_x,grid_y) # Getting indexes where there is no data
           
    # Define random path
    np.random.seed(seed) # Seeding
    randpath = np.random.permutation(non_duplicate) # randomize this array
    
    # Preallocate memory
    sim = np.zeros([np.size(randpath),3])
    
    
    for ii in range(np.size(randpath)):
       # print(randpath[ii])
        sim[ii,0:2] = [grid_x[randpath[ii]], grid_y[randpath[ii]]] # Current point
        
        ind_neigh = get_index_to_nearest_neighbours(kriging_points[:, 0],kriging_points[:, 1],[(sim[ii,0],sim[ii,1])],k=k) # Get neighbours for current point
        #print(ind_neigh)
        OK = OrdinaryKriging(kriging_points[ind_neigh, 0], kriging_points[ind_neigh, 1], kriging_points[ind_neigh, 2], variogram_model='gaussian',
                     verbose=False, enable_plotting=False) # Ordinary kriging 
    
        
        sim[ii,2], ss = OK.execute('grid', sim[ii,0],sim[ii,1]) # Get value for point
    
        #kriging_points = np.concatenate((data_sparse[:,0:3],sim),axis=0)
        kriging_points = np.append(kriging_points,sim[ii,0:3]).reshape((-1,3))
        #print(kriging_points)
    return sim,kriging_points

def example(k=10,Nsim = 4):
    # Create some data for testing
    Nd = 2000    
    Nd_sparse = int(Nd/10)
    
    d = np.ones(Nd)
    d[0:400] = np.random.rand(400)+1
    d[400:800] = np.random.rand(400)+2
    d[800:1200] = np.random.rand(400)+3
    d[1200:1600] = np.random.rand(400)+2
    d[1600:2000] = np.random.rand(400)+1
    
    
    
    #### 1. Define a grid
    Nx = 50
    Ny = 40
    
    x = np.arange(0,Nx)
    y = np.arange(0,Ny)
    
    xx,yy = np.meshgrid(x,y)
    
    
    # Initialize data structure
    data = np.zeros([Nd,3])
    data_sparse = np.zeros([Nd_sparse,4])
    
    # Fill in data structure
    data[:,0] = xx.flatten()
    data[:,1] = yy.flatten()
    data[:,2] = np.copy(d)
    
    
    # Take out 100 random points for sparse 
    np.random.seed(2)
    data_sparse[:,0:3] = data[np.random.choice(Nd,Nd_sparse, replace=False),:]
    
    
    # 2. Place the z-transformed initial data into the nearest grid cells
    data_sparse[:,3] = sp.stats.zscore(data_sparse[:,2]) # Calculate z-score
    
    data_x = data_sparse[:,0]
    data_y = data_sparse[:,1]
    data_val = data_sparse[:,2]
    

    sim = [[]]*Nsim
    # k: Number of nearest neighbours to be included in kriging
    for isim in range(Nsim):
        print('Running sim' + str(isim) + '.......', end="\r")
        rfuncs.tic()
        sim[isim],kriging_points = sgs(data_x,data_y,data_val,xx,yy,seed=isim+1,k=k)
        print('DONE!')
        rfuncs.toc()
    
    
    # Calculate cdfs
    num_bins = 10
    cdf_data,bin_edges = calculate_cdf(data[:,2],num_bins=num_bins)
    cdf_data_sparse,bin_edges_sparse = calculate_cdf(data_sparse[:,2],num_bins=num_bins)
    cdf_data_z_sparse,bin_edges_z_sparse = calculate_cdf(data_sparse[:,3],num_bins=num_bins)
    
    
    
    def plotsim(sim,clim):
        plt.scatter(sim[:,0],sim[:,1],c=sim[:,2],marker='.',s=100)
        plt.clim(clim)
        plt.scatter(data_sparse[:,0],data_sparse[:,1],c=data_sparse[:,2],marker='.',s=100,edgecolor='k')
        plt.clim(clim)
        plt.colorbar()
        plt.xlim([-1,Nx])
        plt.ylim([-1,Ny])
        
    def plotdata(data,clim):
        plt.scatter(data[:,0],data[:,1],c=data[:,2],marker='.',s=100)
        plt.clim(clim)
        plt.colorbar()
        plt.xlim([-1,Nx])
        plt.ylim([-1,Ny])
    
    clim = [1,5]
    
    # Figure of testdata
    plt.close(1)
    plt.figure(1)
    plt.subplot(2,3,1)
    plotdata(data,clim)
    plt.title('Full')
    
    x_cdf= bin_edges[1:]-(bin_edges[1]-bin_edges[0])/2
     
    plt.subplot(2,3,4)
    plt.plot(x_cdf,cdf_data,label='cdf_normalized')
    plt.plot(x_cdf,1-cdf_data,label='cdfinv_normalized')
    plt.grid()
    plt.title('cdf')
    plt.legend()
    
    
    plt.subplot(2,3,2)
    plotdata(data_sparse,clim)
    plt.title('Sparse')
    
    x_cdf= bin_edges_sparse[1:]-(bin_edges_sparse[1]-bin_edges_sparse[0])/2
    
    plt.subplot(2,3,5)
    plt.plot(x_cdf,cdf_data_sparse,label='cdf_normalized')
    plt.plot(x_cdf,1-cdf_data_sparse,label='cdfinv_normalized')
    plt.grid()
    plt.title('cdf')
    plt.legend()
    
    
    plt.subplot(2,3,3)
    plt.scatter(data_sparse[:,0],data_sparse[:,1],c=data_sparse[:,3],marker='.',s=100)
    plt.clim([1,5])
    plt.xlim([-1,Nx])
    plt.ylim([-1,Ny])
    plt.colorbar()
    plt.clim([1,4])
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
    
    
    
    plt.close(2)
    plt.figure(2)
    for i in range(Nsim):
        plt.subplot(2,Nsim/2,i+1)
        plotsim(sim[i],clim)
        plt.title('SIM '+ str(i))
    rfuncs.plot_on_secondary_monitor(1)

        
