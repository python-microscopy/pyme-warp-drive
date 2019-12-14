"""
buffers_cu.py handles importing the CUDA kernels as strings from their respective .cu files and then compiling them
into PyCUDA functions. This allows the same .cu files to be used by others writing functions for MATLAB, etc..

David Baddeley, Andrew Barentine
"""

import os

dPath = os.path.dirname(__file__)

#open and read in sources on module load so we don't redo it every time we need to
#generate a module
with open(os.path.join(dPath,'percentile_buffer.cu'), 'r') as f:
    BUFFER_SOURCE = f.read()


def percentile_buffer():
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#
    # This file contains the CUDA code (to be compiled by PyCUDA) used for percentile
    # -based background estimation
    #-------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------#

    from pycuda.compiler import SourceModule

    # no_extern_c required to avoid compilation error if using modules like thrust
    mod = SourceModule(BUFFER_SOURCE, no_extern_c=True)
    return mod