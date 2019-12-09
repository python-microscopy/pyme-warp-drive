
#include "math.h"

__global__ void var_e_to_invvar_adu(float *var_e, float *inv_var_adu, const float electrons_per_count)
/*
    Convert units of variance from electrons [e-^2] to inverse analog-digital-units [ADU^-2]

    Parameters
    ----------
    var_e: variance map [e-^2]
    inv_var_adu: memory allocation for inverse variance map [ADU^-2]

    CUDA indexing
    -------------
    block
        x: n_columns
            size[1] of the variance map
    grid
        x: n_rows
            size[0] of the variance map
*/
{
    int ind = blockIdx.x * colsize + threadIdx.x;
    inv_var_adu[ind] = 1 / (var_e / (electrons_per_count * electrons_per_count));
}

__global__ void estimate_noise_standard_deviation(float * data, float *readnoise_var,
const float noise_factor, const float electrons_per_count, const float em_gain, float *sigma)
/*
    Estimate a per-pixel noise standard deviation.

    Parameters
    ----------
    data: input data [ADU]
    readnoise_var: (per-pixel) variance due to readout noise [e-^2]
    noise_factor: typically 1.4 for EMCCD when gain is > 10, 1 for CCD and CMOS. [pe-^(-1/2)] see doi: 10.1109/TED.2003.813462
    electrons_per_count: conversion factor between ADU and electrons
    em_gain: conversion factor between electrons post-electron-multiplication and raw photoelectrons. [e-/pe-]
    sigma: memory allocation for per-pixel noise standard deviation [e-]

    CUDA indexing
    -------------
    block
        x: n_columns
            size[1] of the variance map
    grid
        x: n_rows
            size[0] of the variance map

    Notes
    -----
    Note that the PYME.remFitBuf.fitTask.calcSigma returns variance in [ADU^2] while here we return in e-^2
*/
{
    int ind = blockIdx.x * colsize + threadIdx.x;
    sigma[ind] = sqrt(readnoise_var[ind]
                      + noise_factor * noise_factor * electrons_per_count * em_gain * fmaxf(data[ind], 1.0)
                      + em_gain * em_gain / electrons_per_count);
}
