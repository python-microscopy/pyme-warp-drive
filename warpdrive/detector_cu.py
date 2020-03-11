"""
detector_cu.py handles importing the CUDA kernels as strings from their respective .cu files and then compiling them
into PyCUDA functions. This allows the same .cu files to be used by others writing functions for MATLAB, etc..

David Baddeley - Spring 2016
"""

import os

dPath = os.path.dirname(__file__)

#open and read in sources on module load so we don't redo it every time we need to compile
with open(os.path.join(dPath,'filter.cu'), 'r') as f:
    FILTER_SOURCE = f.read()

with open(os.path.join(dPath,'find_peaks.cu'), 'r') as f:
    FINDPEAKS_SOURCE = f.read()

with open(os.path.join(dPath,'gauss_mle.cu'), 'r') as f:
    GAUSSMLE_SOURCE = f.read()


def compile_filters():
    # compile filtering operations used in molecule detection
    from pycuda.compiler import SourceModule
    mod = SourceModule(FILTER_SOURCE)
    return mod

def compile_find_peaks():
    # compile peak-detection functions
    from pycuda.compiler import SourceModule
    mod = SourceModule(FINDPEAKS_SOURCE)
    return mod

def compile_gauss_mle():
    # compile the actual fit
    from pycuda.compiler import SourceModule
    mod = SourceModule(GAUSSMLE_SOURCE, include_dirs=[os.path.join(dPath,'mle_cuda_utils')],
                       options=["--maxrregcount=32"])
    return mod
