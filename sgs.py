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

def remove_duplicate_coordinates_in_array(arr):
    import pandas as pd
    df = pd.DataFrame(arr)
    unique_arr = df.drop_duplicates(subset=[0,1]).values #To return an array rather than a DataFrame object
    return unique_arr

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

###### OBSOLETE?, slow and perhaps wrong!  #####
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

def get_index_for_data_on_grid2(data,xx,yy):
    coordinates = np.array([xx.flatten(),yy.flatten()]).transpose()
    coordind_data = np.zeros(len(data))

    for i in range(len(data)):
        ind = (coordinates == data[i,0:2]).all(axis=1).nonzero()[0]
        coordind_data[i] = ind
    coordind_data = np.sort(coordind_data).astype('int16')
    coordind_no_data = np.delete(np.arange(0,len(coordinates),1).astype('int16'),coordind_data)
    return coordind_no_data,coordind_data

def get_index_to_nearest_neighbours(data_x,data_y,pt,k=10):
    import scipy as sp
    import numpy as np

    data_coords = np.array([data_x,data_y]).transpose()
    hej = sp.spatial.distance.cdist(data_coords,pt,'euclidean')
    indexlist = np.argsort(hej,axis=0)[0:k] # Sort indexes of list 
    return indexlist # Take k nearest neigbours

# 3. Visit the rest of the grid cells at random and perform kriging using all of the data available
def sgs(data_x,data_y,data_val,xx,yy,var_params,var_mod,coordind_no_data=0,seed=1,k=10):
    from IPython.display import clear_output
    print('Setting up random path....')
    kriging_points = np.array([data_x,data_y,data_val]).transpose()

    grid_x = xx.flatten()
    grid_y = yy.flatten()
    
    if isinstance(coordind_no_data, np.ndarray):
        print('List of non-duplicates exists!')
    else:
        print('Calculating list of non-duplicates, be patient!')
        coordind_no_data,coordind_data = get_index_for_data_on_grid2(kriging_points[:,0:2],grid_x,grid_y) # Getting indexes where there is no data
                    
       
    # Define random path
    np.random.seed(seed) # Seeding
    rpath = np.random.permutation(coordind_no_data.shape[0]) # Creating a random permutation of indexes
    non_dup_perm = coordind_no_data[rpath]
    
    # Preallocate memory
    sim = np.zeros([np.size(rpath),3])
    
    
    for ii in range(np.size(rpath)):

       # print(randpath[ii])
        sim[rpath[ii],0:2] = [grid_x[non_dup_perm[ii]], grid_y[non_dup_perm[ii]]] # Current point
        
        ind_neigh = get_index_to_nearest_neighbours(kriging_points[:, 0],kriging_points[:, 1],[(sim[rpath[ii],0],sim[rpath[ii],1])],k=k) # Get neighbours for current point
        #print(ind_neigh)
        OK = OrdinaryKriging(kriging_points[ind_neigh, 0], kriging_points[ind_neigh, 1], 
                             kriging_points[ind_neigh, 2], variogram_model=var_mod,
                             variogram_parameters = var_params,
                             verbose=False, enable_plotting=False) # Ordinary kriging 
    
        try:
            sim[rpath[ii],2], ss = OK.execute('grid', sim[rpath[ii],0],sim[rpath[ii],1]) # Get value for point
        except:
            sim[rpath[ii],2] = 'NaN'
    
        #kriging_points = np.concatenate((data_sparse[:,0:3],sim),axis=0)
        kriging_points = np.append(kriging_points,sim[rpath[ii],0:3]).reshape((-1,3))
        #print(kriging_points)
        if np.mod(ii,500) == 499:
            clear_output(wait=True)
            print('Current progress:',np.round(ii/np.size(rpath)*100,2),'%')
        
#        if np.mod(ii,100) == 99:
#            print(str(round(ii/np.size(randpath)*100,2))+'%')
    print(np.count_nonzero(np.isnan(sim)),'values could not be simulated')
    return sim,kriging_points,rpath,coordind_no_data

def example(k=10,Nsim = 4,var_params = [1,10,0],var_mod = 'spherical'):
    """
    Run example with k conditional points (k=10) and Nsim number of simulations (Nsim=4) 
    Specify: var_params as list [Sill,range,nugget]
             var_mod: type of variogram model (default = 'spherical')
    """
    # Create some data for testing
    
    Nd = 1000    
    Nd_sparse = int(Nd/6)
    
    d = np.ones(Nd)
    d[0:200] = np.random.rand(200)+1
    d[200:400] = np.random.rand(200)+2
    d[400:600] = np.random.rand(200)+3
    d[600:800] = np.random.rand(200)+2
    d[800:1000] = np.random.rand(200)+1
    
    
    
    #### 1. Define a grid
    Nx = 25
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
    

    coordind_no_data,coordind_data = get_index_for_data_on_grid2(np.array([data_x,data_y,data_val]).transpose(),xx,yy) # Getting indexes where there is no data


    sim = [[]]*Nsim
    # k: Number of nearest neighbours to be included in kriging
    for isim in range(Nsim):
        print('Running sim' + str(isim) + '.......', end="\r")
        rfuncs.tic()
        sim[isim],kriging_points,_,coordind_no_data = sgs(data_x,data_y,data_val,xx,yy,seed=isim+1,coordind_no_data=coordind_no_data,k=k,var_params=var_params,var_mod=var_mod)
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

        
