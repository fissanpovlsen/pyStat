import matplotlib.pyplot as plt
import numpy as np

def compute_mod_err(D_delta):
    """ 
    Computes modeling error as as an approximative Gaussian distribution
    from a sample in the form of N realizations, from the (unknown)
    probability distribution that describes the modeling error
    Input: D_delta
    output: C_Tapp, d_Tapp
    """
    N = np.size(D_delta,1) # Number of samples
    d_Tapp = np.sum(D_delta,1)/N
    return d_Tapp 

