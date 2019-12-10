"""
source_prepare.py handles importing the CUDA kernels as strings from their respective .cu files and then compiling them
into PyCUDA functions. This allows the same .cu files to be used by others writing functions for MATLAB, etc..

David Baddeley, Andrew Barentine
"""

import os

dPath = os.path.dirname(__file__)

#open and read in sources on module load so we don't redo it every time we need to generate a module
with open(os.path.join(dPath,'prepare.cu'), 'r') as f:
    BUFFER_SOURCE = f.read()


def prepare():

    from pycuda.compiler import SourceModule

    return SourceModule(BUFFER_SOURCE)