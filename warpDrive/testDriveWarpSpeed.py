from warpDrive import *
import numpy as np
import scipy.io
import time

'''Load data, gain, and variance from .mat files'''
rawdat = np.ascontiguousarray(scipy.io.loadmat('TestData/imqd1_300.mat')['imqd1_300'], dtype=np.float32)
#rawdat = rawdat.astype(np.float32)
#varmap = np.ascontiguousarray(scipy.io.loadmat('/home/aeb85/PycharmProjects/candidatedetection/TiffsForMatlabPyCUDAcomparison/quadrant1/LargeKernel_12pix/varmap.mat')['varmap'][:,:,299], dtype=np.float32)
varmap = np.ascontiguousarray(scipy.io.loadmat('TestData/varmap.mat')['varmap'], dtype=np.float32)
gainmap = np.ascontiguousarray(scipy.io.loadmat('TestData/gainim.mat')['gainim'], dtype=np.float32)


halfFilt = np.int32(6) #set half of the filter size
dfilter1 = normUnifFilter(2*halfFilt)
halfFilt2 = np.int32(round(0.5*halfFilt))
dfilter2 = normUnifFilter(2*halfFilt2)

dshape = (200, 200)
ditemsize = 4  #each pixel is a float32


'''Generate warpDrive'''
_warpDrive = detector(np.shape(rawdat), rawdat.dtype.itemsize, dfilter1, dfilter2)

_warpDrive.allocateMem()
_warpDrive.prepvar(varmap, gainmap)

stppwr = 0
#stppwr = 0
print 10**stppwr
#cuda.start_profiler()


telapsed = []
for ind in range(0, 10**stppwr):
    t0 = time.time()
    _warpDrive.smoothFrame(rawdat)
    _warpDrive.getCand(3.7, 16) #adjusted threshold in order to run the same nubmer of fits as Matlab. roi16 sets maxfilt size = 15
    _warpDrive.fitItSlow(18) #NOTE: need getCand run each loop in order to test fitItSlow because candPos is zerod after fit each run
    t1 = time.time()
    telapsed.append(t1-t0)

#chan1.offIt()
#chan2.offIt()

################################################################################
################################################################################
################################################################################
################################################################################

print('Mean runtime of %.7f, +- %.7f' % (np.mean(telapsed), np.std(telapsed)))

print _warpDrive.dpars.shape

import matplotlib.pyplot as plt
plt.imshow(rawdat)
plt.scatter(_warpDrive.dpars[:, 0], _warpDrive.dpars[:, 1])
plt.show()