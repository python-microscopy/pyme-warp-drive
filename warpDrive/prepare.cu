
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
    int ind = blockIdx.x * gridDim.x + threadIdx.x;
    inv_var_adu[ind] = 1.0f / (var_e[ind] / (electrons_per_count * electrons_per_count));
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
    int ind = blockIdx.x * gridDim.x + threadIdx.x;
    sigma[ind] = sqrt(readnoise_var[ind]
                      + noise_factor * noise_factor * electrons_per_count * em_gain * fmaxf(data[ind], 1.0f)
                      + em_gain * em_gain / electrons_per_count);
}

__device__ float d_estimate_noise_standard_deviation(float data, float readnoise_var, const float noise_factor,
const float electrons_per_count, const float em_gain)
/*
    Estimate a per-pixel noise standard deviation.

    Parameters
    ----------
    data: input data [e-]
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
                + noise_factor * noise_factor * em_gain * fmaxf(data, 1.0f)
                + em_gain * em_gain / electrons_per_count);
}


__global__ void raw_adu_to_e_and_estimate_noise(float *data, float *readnoise_var, float *darkmap, float *flatmap,
const float noise_factor, const float electrons_per_count, const float em_gain, float *sigma)
/*
    Estimates a per-pixel noise standard deviation and converts a raw ADU frame into units of e-. The input data frame
    is modified in-place.

    Parameters
    ----------
    data: input data [ADU]
    readnoise_var: (per-pixel) variance due to readout noise [e-^2]
    darkmap: per-pixel map of the analog-digital offset [ADU]
    flatmap: flatfield map [unitless]
    noise_factor: typically 1.4 for EMCCD when gain is > 10, 1 for CCD and CMOS. [pe-^(-1/2)] see doi: 10.1109/TED.2003.813462
    electrons_per_count: conversion factor between ADU and electrons
    em_gain: conversion factor between electrons post-electron-multiplication and raw photoelectrons. [e-/pe-]
    sigma: memory allocation for per-pixel noise standard deviation [e-]

    CUDA indexing
    -------------
    Max number of threads per block is 1024 for pretty much all cards. Memory is row-major, so we coalesce if we use the
    1-d block to read along the row of the image, and use the block index to read along columns. Note that the maximum
    size of data is limited to [1024, 2^31 - 1] for pretty much all cards.
    block
        x: data.shape[0]
    grid
        x: data.shape[1]

    Notes
    -----
    Note that the PYME.remFitBuf.fitTask.calcSigma returns variance in [ADU^2] while here we return in e-^2
*/
{
    int ind = blockIdx.x * gridDim.x + threadIdx.x;
    // camera-correct data, and convert units
    data[ind] = (data[ind] - darkmap[ind]) * flatmap[ind] * electrons_per_count;  // [ADU] -> [e-]

    // estimate noise
    sigma[ind] = d_estimate_noise_standard_deviation(data[ind], readnoise_var[ind], noise_factor, electrons_per_count,
                                                   em_gain);
}

__global__ void correct_frame_and_convert_adu_to_e(float *data, float *darkmap, float *flatmap,
const float electrons_per_count)
/*
    Camera-corrects input data and converts from analog-digital units to e-. The input data frame is modified in-place.

    Parameters
    ----------
    data: input data [ADU]
    darkmap: per-pixel map of the analog-digital offset [ADU]
    flatmap: flatfield map [unitless]
    electrons_per_count: conversion factor between ADU and electrons

    CUDA indexing
    -------------
    Max number of threads per block is 1024 for pretty much all cards. Memory is row-major, so we coalesce if we use the
    1-d block to read along the row of the image, and use the block index to read along columns. Note that the maximum
    size of data is limited to [1024, 2^31 - 1] for pretty much all cards.
    block
        x: data.shape[0]
    grid
        x: data.shape[1]
*/
{
    int ind = blockIdx.x * gridDim.x + threadIdx.x;
    // camera-correct data, and convert units
    data[ind] = (data[ind] - darkmap[ind]) * flatmap[ind] * electrons_per_count;  // [ADU] -> [e-]
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
        x: n_rows
            size[0] of the variance map
    grid
        x: n_columns
            size[1] of the variance map

    Notes
    -----
    Gain [ADU / e-] is the flatfield [unitless] divided by electrons_per_count [e-/ADU]. Taking the variance to be in
    ADU^2, the variance / gain^2 term appears to have units of [e-^2] rather than [e-^2]. Presumably one factor of [e-]
    can be ignored due to the expectation and variance of a Poisson distribution both being the rate constant.
*/
{
    int ind = blockIdx.x * gridDim.x + threadIdx.x;

    // d - o / g = (d - o) * flatfield * electrons_per_count
    variance_over_gain_squared[ind] = readnoise_var[ind] * powf(flatmap[ind] * electrons_per_count, 2.0f);
}

