#!/usr/bin/python

from pylab import *
import pyMLEfit

x = (1 + 50*exp(-(((mgrid[:17.] - 7.5)**2)[:,None] + ((mgrid[:17.] - 6)**2)[None,:])/8)[:,:,None]*np.ones(10000)[None,None,:]).astype('f')
varim = np.ones_like(x);gainim = np.ones_like(x)

for i in range(1):
	o = pyMLEfit.doMLEFits(x, 2., 50, 1, varim, gainim)