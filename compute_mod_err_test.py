
import matplotlib.pyplot as plt
plt.close('all')
import numpy as np


from compute_mod_err import compute_mod_err


N = 1000 # number of samples
M = 50 # number of modelparameters
a = np.ones([M,N])
np.random.seed(seed=1)
b = np.random.rand(M,N)

test = a+b


d_Tapp = compute_mod_err(test)

hej = np.transpose(d_Tapp)

D_Tapp = np.transpose(np.tile(d_Tapp, (N,1)))
D_diff = test-D_Tapp
C_Tapp = np.dot(D_diff,np.transpose(D_diff))/N


plt.figure(1)
plt.imshow(D_Tapp, interpolation='none',cmap='viridis')
plt.colorbar()
plt.show()
#plt.get_current_fig_manager().window.raise_()
cfm = plt.get_current_fig_manager()
cfm.window.activateWindow()
cfm.window.raise_()

plt.figure(2)
plt.imshow(D_diff, interpolation='none',cmap='viridis')
plt.colorbar()
plt.show()
#plt.get_current_fig_manager().window.raise_()
cfm = plt.get_current_fig_manager()
cfm.window.activateWindow()
cfm.window.raise_()

plt.figure(3)
plt.imshow(C_Tapp, interpolation='none',cmap='viridis')
plt.colorbar()
plt.show()
#plt.get_current_fig_manager().window.raise_()
cfm = plt.get_current_fig_manager()
cfm.window.activateWindow()
cfm.window.raise_()