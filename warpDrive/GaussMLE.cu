/*#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#
# gaussMLE_Fang implements an astigmatic gaussian maximum-likelihood estimator
# fit in CUDA. This code was written by Fang Huang, converted to a pixel-wise
# approach by David Baddeley, and then merged to Bewersdorf/Baddeley Labs'
# PyCUDA localization routine by AESB.
# AESB 03/31/2016
#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#*/
// [Parameters CRLBs LL]=GPUgaussMLE(data,PSFSigma,iterations)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "MatInvLib.h"
#include "GPUgaussLib.cuh"
#include "GPUgaussMLEv2.h"

#define pi 3.141592f
#define NUM_VARS 4
#define NUM_VARS_ASTIG 6
#define BLOCK_MAX_SIZE 512

__device__ void kernel_sum_reduce(const int BlockSize, const int pixelIndex, volatile float *A, volatile float *Aout) {
    /* Calculate the sum by performing cascaded sums across threads

    */

    //unrolled loop to SUM across threads in O(log2(N))
    //first sum bottom half with top half,
    //then repeat with succesively smaller folds
    //concept borrowed/adapted from reduction code in cuda toolkit examples
    __syncthreads();

    //We expect our block size to be between around 120 and 441
    //pixels. First two rows are thus somewhat of a special case
    if ((BlockSize >= 256) && (pixelIndex < (BlockSize -256))){
        A[pixelIndex] += A[pixelIndex + 256];
    }

    __syncthreads();

    // This has to work as a 2nd iteration for a block size > 256 AND
    // as a first iteration for a block size between 128 and 256.
    if ((BlockSize >= 128) && (pixelIndex < min(128, BlockSize - 128))){
        A[pixelIndex] += A[pixelIndex + 128];
    }

    __syncthreads();

    if ((BlockSize >= 64) && (pixelIndex < min(64, BlockSize - 64))){
        A[pixelIndex] += A[pixelIndex + 64];
    }

    __syncthreads();

    //From here on in, we know that our BlockSize test is going to pass
    // so we can remove it

    //We're also just acting on one warp (<32 threads), so we a) don't need __syncthreads
    // and b) don't gain anything by further restricting the number of threads

    if ((pixelIndex < 32)){
        A[pixelIndex] += A[pixelIndex + 32];
        A[pixelIndex] += A[pixelIndex + 16];
        A[pixelIndex] += A[pixelIndex + 8];
        A[pixelIndex] += A[pixelIndex + 4];
        A[pixelIndex] += A[pixelIndex + 2];
    }

    //__syncthreads();

    // do the final sum and copy into output variable
    if (pixelIndex == 0){
        *Aout = (A[0] + A[1]);
    }

    __syncthreads();

    return;
}

__device__ void kernel_min_reduce(const int BlockSize, const int pixelIndex, volatile float *A, volatile float *Aout) {
    /* Calculate the minimum by performing cascaded sums across threads

    */

    //unrolled loop to take minumum across threads in O(log2(N))
    //first sum bottom half with top half,
    //then repeat with succesively smaller folds
    __syncthreads();

    //We expect our block size to be between around 120 and 441
    //pixels. First two rows are thus somewhat of a special case
    if ((BlockSize >= 256) && (pixelIndex < (BlockSize -256))){
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 256]);
    }

    __syncthreads();

    // This has to work as a 2nd iteration for a block size > 256 AND
    // as a first iteration for a block size between 128 and 256.
    if ((BlockSize >= 128) && (pixelIndex < min(128, BlockSize - 128))){
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 128]);
    }

    __syncthreads();

    if ((BlockSize >= 64) && (pixelIndex < min(64, BlockSize - 64))){
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 64]);
    }

    __syncthreads();

    //From here on in, we know that our BlockSize test is going to pass
    // so we can remove it

    //We're also just acting on one warp, so we a) don't need __syncthreads
    // and b) don't gain anything by further restricting the number of threads

    if ((pixelIndex < 32)){
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 32]);
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 16]);
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 8]);
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 4]);
        A[pixelIndex] = min(A[pixelIndex], A[pixelIndex + 2]);
    }

    //__syncthreads();

    if (pixelIndex == 0){
        *Aout = min(A[0], A[1]);
    }

    __syncthreads();

    return;
}

__device__ void kernel_max_reduce(const int BlockSize, const int pixelIndex, volatile float *A, volatile float *Aout) {
    /* Calculate the maximum by performing cascaded sums across threads

    */

    //unrolled loop to take max across threads in O(log(N))
    //first sum bottom half with top half,
    //then repeat with succesively smaller folds
    __syncthreads();

    //We expect our block size to be between around 120 and 441
    //pixels. First two rows are thus somewhat of a special case
    if ((BlockSize >= 256) && (pixelIndex < (BlockSize -256))){
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 256]);
    }

    __syncthreads();

    // This has to work as a 2nd iteration for a block size > 256 AND
    // as a first iteration for a block size between 128 and 256.
    if ((BlockSize >= 128) && (pixelIndex < min(128, BlockSize - 128))){
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 128]);
    }

    __syncthreads();

    if ((BlockSize >= 64) && (pixelIndex < min(64, BlockSize - 64))){
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 64]);
    }

    __syncthreads();

    //From here on in, we know that our BlockSize test is going to pass
    // so we can remove it

    //We're also just acting on one warp, so we a) don't need __syncthreads
    // and b) don't gain anything by further restricting the number of threads

    if ((pixelIndex < 32)){
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 32]);
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 16]);
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 8]);
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 4]);
        A[pixelIndex] = max(A[pixelIndex], A[pixelIndex + 2]);
    }

    //__syncthreads();

    if (pixelIndex == 0){
        *Aout = max(A[0], A[1]);
    }

    __syncthreads();

    return;
}

__device__ void kernel_sum_reduce_dual(const int BlockSize, const int pixelIndex, volatile float *A, volatile float *B, volatile float *Aout, volatile float *Bout) {
    /* Calculate the sum by performing cascaded sums across threads

    Sums two variables (A and B) at once to maximize throughput

    */

    //unrolled loop to SUM across threads in O(log2(N))
    //first sum bottom half with top half,
    //then repeat with succesively smaller folds
    __syncthreads();

    //We expect our block size to be between around 120 and 441
    //pixels. First two rows are thus somewhat of a special case
    if ((BlockSize >= 256) && (pixelIndex < (BlockSize -256))){
        A[pixelIndex] += A[pixelIndex + 256];
        B[pixelIndex] += B[pixelIndex + 256];
    }

    __syncthreads();

    // This has to work as a 2nd iteration for a block size > 256 AND
    // as a first iteration for a block size between 128 and 256.
    if ((BlockSize >= 128) && (pixelIndex < min(128, BlockSize - 128))){
        A[pixelIndex] += A[pixelIndex + 128];
        B[pixelIndex] += B[pixelIndex + 128];
    }

    __syncthreads();

    if ((BlockSize >= 64) && (pixelIndex < min(64, BlockSize - 64))){
        A[pixelIndex] += A[pixelIndex + 64];
        B[pixelIndex] += B[pixelIndex + 64];
    }

    __syncthreads();

    //From here on in, we know that our BlockSize test is going to pass
    // so we can remove it

    //We're also just acting on one warp, so we a) don't need __syncthreads
    // and b) don't gain anything by further restricting the number of threads

    if ((pixelIndex < 32)){
        A[pixelIndex] += A[pixelIndex + 32];
        B[pixelIndex] += B[pixelIndex + 32];

        A[pixelIndex] += A[pixelIndex + 16];
        B[pixelIndex] += B[pixelIndex + 16];

        A[pixelIndex] += A[pixelIndex + 8];
        B[pixelIndex] += B[pixelIndex + 8];

        A[pixelIndex] += A[pixelIndex + 4];
        B[pixelIndex] += B[pixelIndex + 4];

        A[pixelIndex] += A[pixelIndex + 2];
        B[pixelIndex] += B[pixelIndex + 2];
    }

    //__syncthreads();

    if (pixelIndex == 0){
        *Aout = (A[0] + A[1]);
        *Bout = (B[0] + B[1]);
    }

    __syncthreads();

    return;
}


__global__ void kernel_MLEFit_pix_threads_astig(float *d_data, float PSFSigma, int sz, int iterations,
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB,
        int *candPos, int numbCol, int candOff){ //, float *testROI)
    /* A version of MLEFit that uses per-pixel, rather than per fit threads

    Each block consists corresponds to one ROI. threadIdx.x is the x pixel coordinate,
    threadIdx.y is the y coordinate.

    David Baddeley
    Feb 2016
    Modified to accept position list of candidate molecules and crop ROIs from data already present on GPU
    AESB Apr 2016

    */


    //these will store the data for the pixel being processed
    float pixel_data;
    float pixel_variance;
    float pixel_gain;
    float pixel_variance_over_gain_squared;

    //allocate arrays for matrices
    float M[NUM_VARS_ASTIG*NUM_VARS_ASTIG];
    __shared__ float Diag[NUM_VARS_ASTIG];
    float Minv[NUM_VARS_ASTIG*NUM_VARS_ASTIG];

    //One block = one ROI
    int BlockSize = blockDim.x*blockDim.y;
    //int blockOffset = BlockSize*blockIdx.x; //the offset in memory to the current ROI
    int pixelIndex = threadIdx.x + blockDim.x*threadIdx.y; //index of pixel in ROI



    /*if (pixelIndex == 0){
        printf("blockIdx.x: %d\n", blockIdx.x);
        printf("BlockSize: %d\n", BlockSize);
        printf("blockOffset: %d\n", blockOffset);
    }*/

    //float gain=0.45;
    //int ii, jj, kk, ll;
    int kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;

    volatile __shared__ float data_sum;

    //Allocate arrays for the gradients etc ...
    float dudt[NUM_VARS_ASTIG];
    float d2udt2[NUM_VARS_ASTIG];

    //two temporary arrays which are used whenever we want to reduce (sum, min, max)
    //across threads
    volatile __shared__ float s_temp0[BLOCK_MAX_SIZE];
    volatile __shared__ float s_temp1[BLOCK_MAX_SIZE];

    //The numerator and denominator for the update step. These are used from multiple threads, and need
    //to be shared
    volatile __shared__ float NR_Numerator[NUM_VARS_ASTIG];
    volatile __shared__ float NR_Denominator[NUM_VARS_ASTIG];

    //the parameter vector
    volatile __shared__ float theta[NUM_VARS_ASTIG];

    //can these be const??? I don't see maxjump being revised anywhere
    const float maxjump[NUM_VARS_ASTIG]={1e0, 1e0, 1e2, 2e0, 1e-1, 1e-1}; // maxjump is revised such that the intensity can converge to the correct value
    const float gamma[NUM_VARS_ASTIG]={1.0, 1.0, 0.5, 1.0, 1.0, 1.0};

    //variable to store maximum intensity in ROI
    float Nmax;

    //Prevent read/write past end of array
    //if ((bx*BlockSize+tx)>=Nfits) return;

    //zero arrays (each thread zeros one location)
    // As we have more pixels than entries in the matrix, we will only use the first NV*NV threads
    // We've effectively parallized the loop across the threads
    // CHECK - Do we still need to zero M (we now do the sum and then assign, rather than a progressive sum)
    if (pixelIndex < NUM_VARS_ASTIG*NUM_VARS_ASTIG){
        M[pixelIndex] = 0;
        Minv[pixelIndex] = 0;
    }


    //maybe put a __syncthreads here to ensure following ops are coallesced??
    __syncthreads(); //Is this necessary?

    //load data from global memory (each thread loads the data for it's pixel)
    //as each thread accesses a consecutive memory location, access should be coalesced
    /*
    pixel_data = d_data[blockOffset + pixelIndex];
    pixel_variance = d_varim[blockOffset + pixelIndex];
    pixel_gain = d_gainim[blockOffset + pixelIndex];
    */


    //int uplc = candPos[blockIdx.x] - (0.5*blockDim.x) - (0.5*blockDim.y)*numbCol; //upper left hand corner of the subROI
    //FIXME: Check why we need this offset -> potentially has to do with convolutions using even kernel sizes
    int uplc = candPos[blockIdx.x + candOff] - (0.5*(blockDim.x - 2)) - (0.5*(blockDim.y - 2))*numbCol;
    //pixel_data = d_data[uplc + threadIdx.x + threadIdx.y*numbCol];
    pixel_variance = d_varim[uplc + threadIdx.x + threadIdx.y*numbCol];
    pixel_gain = d_gainim[uplc + threadIdx.x + threadIdx.y*numbCol];
    pixel_data = d_data[uplc + threadIdx.x + threadIdx.y*numbCol]/pixel_gain;

    /*
    int BlockSize = blockDim.x*blockDim.y;
    int blockOffset = BlockSize*blockIdx.x; //the offset in memory to the current ROI
    int pixelIndex = threadIdx.x + blockDim.x*threadIdx.y; //index of pixel in ROI
    */

    //Precompute these quantities (previously re-evaluated on each iteration)
    pixel_variance_over_gain_squared = pixel_variance/(pixel_gain*pixel_gain);
    data=pixel_data + pixel_variance_over_gain_squared;

    //wait untill all loads have taken place
    __syncthreads();


    //if (blockIdx.x ==0){
    //testROI[pixelIndex] = data;
    //__syncthreads();
    //}
    //initial values
    //==============
    //Find Center of mass

    //find sum of all pixels in ROI
    s_temp0[pixelIndex] = pixel_data;
    kernel_sum_reduce(BlockSize, pixelIndex, s_temp0, &data_sum);

    //printf("data_sum: %f", data_sum);

    // Find centroid
    s_temp0[pixelIndex] = pixel_data*threadIdx.x/data_sum;
    s_temp1[pixelIndex] = pixel_data*threadIdx.y/data_sum;

    kernel_sum_reduce_dual(BlockSize, pixelIndex, s_temp0, s_temp1, &theta[0], &theta[1]);

    /*
    if ((pixelIndex == 0) && (blockIdx.x == 0)){
        printf("data_sum: %f ;", data_sum);
        printf("x0, y0: %f, %f ;", theta[0], theta[1]);
        //theta[0] = 3;
        //printf("theta0 = %f", theta[0]);
    }*/

    //find max and min values
    s_temp0[pixelIndex] = pixel_data;
    s_temp1[pixelIndex] = pixel_data;

    kernel_min_reduce(BlockSize, pixelIndex, s_temp0, &theta[3]);
    kernel_max_reduce(BlockSize, pixelIndex, s_temp1, &Nmax);

    if (pixelIndex == 0)
        //We only need to do this once - do it in thread 0
        theta[2]=fmaxf(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);

        theta[4]=PSFSigma;
        theta[5]=PSFSigma;

    __syncthreads();

    for (kk=0;kk<iterations;kk++) {//main iterative loop

        PSFx=kernel_IntGauss1D(threadIdx.x, theta[0], theta[4]);
        PSFy=kernel_IntGauss1D(threadIdx.y, theta[1], theta[5]);

        model=theta[3] + theta[2]*PSFx*PSFy + pixel_variance_over_gain_squared;
        //data=pixel_data + pixel_variance_over_gain_squared; //now calculated outside loop

        //calculating derivatives
        kernel_DerivativeIntGauss1D(threadIdx.x, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
        kernel_DerivativeIntGauss1D(threadIdx.y, theta[1], theta[5], theta[2], PSFx, &dudt[1], &d2udt2[1]);
        kernel_DerivativeIntGauss1DSigma(threadIdx.x, theta[0], theta[4], theta[2], PSFy, &dudt[4], &d2udt2[4]);
        kernel_DerivativeIntGauss1DSigma(threadIdx.y, theta[1], theta[5], theta[2], PSFx, &dudt[5], &d2udt2[5]);
        dudt[2] = PSFx*PSFy;
        d2udt2[2] = 0.0;
        dudt[3] = 1.0;
        d2udt2[3] = 0.0;

        cf=0.0;
        df=0.0;
        if (model>10e-3) cf=data/model-1;
        if (model>10e-3) df=data/(model*model);
        cf=min(cf, 10e4);
        df=min(df, 10e4);

        for (ll=0;ll < NUM_VARS_ASTIG;ll++){
            s_temp0[pixelIndex] = dudt[ll]*cf;
            s_temp1[pixelIndex] = d2udt2[ll]*cf - pow(dudt[ll], 2)*df;

            //sum over all pixels
            kernel_sum_reduce_dual(BlockSize, pixelIndex, s_temp0, s_temp1, &NR_Numerator[ll], &NR_Denominator[ll]);
        }


        // The update
        ll = pixelIndex;
        //rather than looping over the variables, use the first NUM_VARS_ASTIG threads
        if (ll < NUM_VARS_ASTIG){
            if (kk<2) //FIXME check to see if this should be for the first 5 iterations instead
            //if (kk<5) //FIXME check to see if this should be for the first 5 iterations instead
                theta[ll]-=gamma[ll]*min(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
            else
                theta[ll]-=min(fmaxf(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        }
        __syncthreads();

        if (pixelIndex == 0){
            // Do this in the 0th thread only - i.e. once per block/ROI
            // Any other constraints
            theta[2]=fmaxf(theta[2], 1.0);
            theta[3]=fmaxf(theta[3], 0.01);
            theta[4]=fmaxf(theta[4], PSFSigma/10.0);
            theta[5]=fmaxf(theta[5], PSFSigma/10.0);

            /*printf("block: %d, iteration: %d, theta: [%f, %f, %f, %f]\n", blockIdx.x, kk, theta[0], theta[1], theta[2], theta[3]);
            printf("block: %d, iteration: %d, num/dem: [%f/%f, %f/%f, %f/%f, %f/%f]\n", blockIdx.x, kk,
                NR_Numerator[0], NR_Denominator[0], NR_Numerator[1], NR_Denominator[1],
                NR_Numerator[2], NR_Denominator[2], NR_Numerator[3], NR_Denominator[3]
                );*/
        }

        __syncthreads();

    }

    if (calcCRB){
        // Calculating the CRLB and LogLikelihood
        //Div=0.0;

        //evaluate model with final parameters
        //Note that if fit is converged, we could probably skip this step
        //as model, dudt etc ... should still have the correct values from the last fit iteration
        PSFx=kernel_IntGauss1D(threadIdx.x, theta[0], theta[4]);
        PSFy=kernel_IntGauss1D(threadIdx.y, theta[1], theta[5]);

        model=theta[3] + theta[2]*PSFx*PSFy + pixel_variance_over_gain_squared;

        //calculating derivatives
        //kernel_DerivativeIntGauss1D(threadIdx.x, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        //kernel_DerivativeIntGauss1D(threadIdx.y, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss1D(threadIdx.x, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(threadIdx.y, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss1DSigma(threadIdx.x, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
        kernel_DerivativeIntGauss1DSigma(threadIdx.y, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);

        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;

        //Building the Fisher Information Matrix
        for (kk=0;kk<NUM_VARS_ASTIG;kk++) for (ll=kk;ll<NUM_VARS_ASTIG;ll++){

            //reuse s_temp0 here to save memory
            s_temp0[pixelIndex] = dudt[ll]*dudt[kk]/model;

            //sum over all pixels
            kernel_sum_reduce(BlockSize, pixelIndex, s_temp0, &M[kk*NUM_VARS_ASTIG+ll]);

            if (pixelIndex == 0){
                //execute in first thread of block only

                //exploit known matrix symetry and copy value to the mirrored entry
                M[ll*NUM_VARS_ASTIG+kk] = M[kk*NUM_VARS_ASTIG+ll];
            }

            __syncthreads();

        }

        //LogLikelyhood
        if (model>0){
            if (data>0){
                //reuse s_temp0 here to save memory
                s_temp0[pixelIndex] = data*log(model)-model-data*log(data)+data;
            } //Div+=data*log(model)-model-data*log(data)+data;
            else {
                s_temp0[pixelIndex] = -model;
            } //Div+=-model;
        } else s_temp0[pixelIndex] =0;

        //sum over all pixels
        kernel_sum_reduce(BlockSize, pixelIndex, s_temp0, &Div);

        if (pixelIndex == 0){
            //only in first thread ...

            //write our log likelihood back into global memory
            d_LogLikelihood[blockIdx.x] = Div;

            // Matrix inverse (CRLB=F^-1) and output assigments
            kernel_MatInvN(M, Minv, Diag, NUM_VARS_ASTIG);
        }

    }
    __syncthreads();

    //Add the offset back to the fit positions
    //int uplc = candPos[blockIdx.x] - (0.5*blockDim.x) - (0.5*blockDim.y)*numbCol; //upper left hand corner of the subROI
    //pixel_data = d_data[uplc + threadIdx.x + threadIdx.y*numbCol];
    //int xoffSet = uplc % numbCol;
    //int yoffSet = uplc / numbCol;

    //printf("Xuplc : %d ,", (uplc % numbCol));
    //theta[0] += (float) (uplc % numbCol);
    //theta[1] += (float) (uplc / numbCol);


    if (pixelIndex == 0){
        //if (blockIdx.x == 0) printf("x pos %f, y pos %f    ", theta[0], theta[1]);
        //if (blockIdx.x == 0) printf("x Offset %d, y Offset %d    ", uplc % numbCol, uplc / numbCol);
        //printf("Candidate Position: %d, x Offset %d, y Offset %d   ", candPos[blockIdx.x], uplc % numbCol, uplc / numbCol);
        theta[0] += (float) (uplc % numbCol); //Row offset (x)
        theta[1] += (float) (uplc / numbCol); //Column offset (y)
        //rezero the list of candidate molecules so that it no longer needs to be reallocated for each frame's fit.
        candPos[blockIdx.x+candOff] = 0;
    }



    //write to global arrays
    //CHECK!! - are output arrays transposed???
    //We want this matrix orientation as it optimizes memory access speed (coalescing)
    if (pixelIndex < NUM_VARS_ASTIG){
        d_Parameters[NUM_VARS_ASTIG*(blockIdx.x) + pixelIndex] = theta[pixelIndex];
        d_CRLBs[NUM_VARS_ASTIG*(blockIdx.x) + pixelIndex] = Diag[pixelIndex];
    }




    return;
}
