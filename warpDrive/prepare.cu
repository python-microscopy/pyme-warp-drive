
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


__global__ void estimate_noise_standard_deviation(float *data, float *readnoise_var,
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

__device__ float estimate_noise_standard_deviation(const float data, const float readnoise_var,
const float noise_factor, const float electrons_per_count, const float em_gain, const float sigma)
/*
    Estimate a per-pixel noise standard deviation.

    Parameters
    ----------
    data: input data [ADU]
    readnoise_var: (per-pixel) variance due to readout noise [e-^2]
    noise_factor: typically 1.4 for EMCCD when gain is > 10, 1 for CCD and CMOS. [pe-^(-1/2)] see doi: 10.1109/TED.2003.813462
    electrons_per_count: conversion factor between ADU and electrons
    em_gain: conversion factor between electrons post-electron-multiplication and raw photoelectrons. [e-/pe-]

    Returns
    -------
    sigma: per-pixel noise standard deviation [e-]

    Notes
    -----
    Note that the PYME.remFitBuf.fitTask.calcSigma returns variance in [ADU^2] while here we return in e-^2
*/
{
    return sqrt(readnoise_var
                + noise_factor * noise_factor * electrons_per_count * em_gain * fmaxf(data, 1.0)
                + em_gain * em_gain / electrons_per_count);
}


__global__ void correct_frame_and_estimate_noise(float *data, float *readnoise_var, float *darkmap, float *flatmap,
const float noise_factor, const float electrons_per_count, const float em_gain, float *sigma)
/*
    Estimates a per-pixel noise standard deviation and converts a raw ADU frame into units of e-. The input data frame
    is modified in-place.

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
    FIXME - convert this to indexing used in background buffer so we can handle larger frames.

    Notes
    -----
    Note that the PYME.remFitBuf.fitTask.calcSigma returns variance in [ADU^2] while here we return in e-^2
*/
{
    int ind = blockIdx.x * colsize + threadIdx.x;
    // camera-correct data
    data[ind] = (data[ind] - darkmap[ind]) * flatmap[ind];  // [ADU]

    // estimate noise
    sigma[ind] = estimate_noise_standard_deviation(data[ind], readnoise_var[ind], noise_factor, electrons_per_count,
                                                   em_gain);
    // convert data to electrons
    data[ind] *= electrons_per_count;  // [ADU] -> [e-]
}


__global__ void variance_over_gain_squared(float *readnoise_var, float *flatmap, const float electrons_per_count,
float *variance_over_gain_squared)
/*
    precalculate variance / gain^2, a term in the approximation of the addition of a Gaussian (readnoise) and Poisson
    (shot noise) random variables used to model the noise characteristics of an sCMOS camera. See doi:10.1038/nmeth.2488

    Parameters
    ----------
    readnoise_var: variance map [e-^2]
    flatmap: flatfield map [unitless]
    electrons_per_count: conversion factor between e- and ADU
    variance_over_gain_squared: memory allocation to store variance over gain squared (see note on units)

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
    Gain [ADU / e-] is the flatfield [unitless] divided by electrons_per_count [e-/ADU]. Taking the variance to be in
    ADU^2, the variance / gain^2 term appears to have units of [e-^2] rather than [e-^2]. Presumably one factor of [e-]
    can be ignored due to the expectation and variance of a Poisson distribution both being the rate constant.
*/
{
    int ind = blockIdx.x * colsize + threadIdx.x;

    // d - o / g = (d - o) * flatfield * electrons_per_count
    variance_over_gain_squared[ind] = readnoise_var * pow(flatmap * electrons_per_count, 2);
}

