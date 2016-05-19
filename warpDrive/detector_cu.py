import os

dPath = os.path.dirname(__file__)

#open and read in sources on module load so we don't redo it every time we need to
#generate a module
with open(os.path.join(dPath,'filter.cu'), 'r') as f:
    FILTER_SOURCE = f.read()

with open(os.path.join(dPath,'findPeaks.cu'), 'r') as f:
    FINDPEAKS_SOURCE = f.read()

with open(os.path.join(dPath,'GaussMLE.cu'), 'r') as f:
    GAUSSMLE_SOURCE = f.read()


def detectorCompileNBlock_sCMOS():
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#
    # This file contains the CUDA code (to be compiled by PyCUDA) that is used for
    # candidate molecule detection.
    # AESB 03/15/2016
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#

    from pycuda.compiler import SourceModule
    ######################### DEFINE CUDA KERNELS ###################################

    #dPath = os.path.dirname(__file__)

    #with open(os.path.join(dPath,'filter.cu'), 'r') as f:
    #    fitKernel = f.read()


    mod = SourceModule(FILTER_SOURCE))
    return mod

def finderCompile():
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#
    # This file contains the CUDA code (to be compiled by PyCUDA) that is used for
    # CLEAN algorithm
    # AESB 03/15/2016
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#

    from pycuda.compiler import SourceModule
    ######################### DEFINE CUDA KERNELS ###################################
    #with open(os.path.join(dPath,'findPeaks.cu'), 'r') as f:
    #    fitKernel = f.read()


    mod = SourceModule(FINDPEAKS_SOURCE))

    return mod

def gaussMLE_Fang_David():
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#
    # gaussMLE_Fang implements an astigmatic gaussian maximum-likelihood estimator
    # fit in CUDA. This code was written by Fang Huang, converted to a pixel-wise
    # approach by David Baddeley, and then merged to Bewersdorf/Baddeley Labs'
    # PyCUDA localization routine by AESB.
    # AESB 03/31/2016
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#

    from pycuda.compiler import SourceModule
    ######################### DEFINE CUDA KERNELS ###################################
    #dPath = os.path.dirname(__file__)
    #print dPath

    #read the kernel from the .cu file
    #this enables us to a) get propper syntax highlighting for the code and b) use the same code for this and the matlab version
    #with open(os.path.join(dPath,'GaussMLE.cu'), 'r') as f:
    #    fitKernel = f.read()
    
    #print os.path.join(dPath,'cudaResources')
    #mod = SourceModule(fitKernel, include_dirs=['/home/aeb85/PycharmProjects/candidatedetection/cudaResources/SingleMLEsCMOS', '/home/aeb85/PycharmProjects/candidatedetection/cudaResources'],
    #options=["--maxrregcount=32"])
    mod = SourceModule(GAUSSMLE_SOURCE, include_dirs=[os.path.join(dPath,'cudaResources')], options=["--maxrregcount=32"])

    return mod
