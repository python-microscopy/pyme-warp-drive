/*Compile the mex file: (old fashion). VS2008 version does not these steps.
 *
 * nvmex -f Z:\CUDA\nvmexoptsxp64.bat       GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
 * nvmex -f Z:\CUDA\nvmexopts_r2008a.bat    GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
 * nvmex -f Z:\CUDA\nvmexopts_9.bat    GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart -output GPUgaussMLEfangMATLAB
 * nvmex -f Z:\cCode\CUDA\nvmexoptsXP32_VS2008_r2008.bat GPUgaussMLE.cu -IC:\cuda\include -LC:\cuda\lib -lcudart
 * nvmex -f Z:\cCode\CUDA\nvmexoptsxp64_VS2008SDK7.bat GPUgaussMLEf1.cu -IC:\cuda\include -LC:\cuda\lib64 -lcudart
 *
 * [Parameters CRLBs LL]=GPUgaussMLE(data,PSFSigma,iterations)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "mex.h"
#include "cuda_runtime.h"
#include "MatInvLib.h"
#include "GPUgaussLib.cuh"
#include "GPUgaussMLEv2.h"
// Thread block size
#define BSZ 64
//#define BSZ 64
#define MEM 1300
#define IMSZ 11
#define IMSZBIG 21
#define NK 128 //number of blocks to run in each kernel
#define pi 3.141592f
#define NV_P 4
#define NV_PS 5
#define NV_PZ 5
#define NV_PS2 6

#define NUM_VARS 4
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

    if ((BlockSize >= 64) && (pixelIndex < 64)){
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

    if ((BlockSize >= 64) && (pixelIndex < 64)){
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

    if ((BlockSize >= 64) && (pixelIndex < 64)){
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

    if ((BlockSize >= 64) && (pixelIndex < 64)){
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


__global__ void kernel_MLEFit_pix_threads(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB){
    /* A version of MLEFit that uses per-pixel, rather than per fit threads 

    Each block consists corresponds to one ROI. threadIdx.x is the x pixel coordinate, 
    threadIdx.y is the y coordinate.

    David Baddeley
    Feb 2016

    */
    

    //these will store the data for the pixel being processed
    float pixel_data;
    float pixel_variance;
    float pixel_gain;
    float pixel_variance_over_gain_squared;

    //allocate arrays for matrices
    __shared__ float M[NUM_VARS*NUM_VARS];
    __shared__ float Diag[NUM_VARS];
    __shared__ float Minv[NUM_VARS*NUM_VARS];
    
    //One block = one ROI
    int BlockSize = blockDim.x*blockDim.y;
    //#define BlockSize (blockDim.x*blockDim.y)

    int blockOffset = BlockSize*blockIdx.x; //the offset in memory to the current ROI
    int pixelIndex = threadIdx.x + blockDim.x*threadIdx.y; //index of pixel in ROI

    //printf("pixelIndex: %d\n", pixelIndex);

    /*if (pixelIndex == 0){
        printf("blockIdx.x: %d\n", blockIdx.x);
        printf("BlockSize: %d\n", BlockSize);
        printf("blockOffset: %d\n", blockOffset);
    }*/
    
    //float gain=0.45;
    //int ii, jj, kk, ll;
    int kk, ll;
    float model, cf, df, data;
    float tmp;
    float PSFy, PSFx;

    volatile __shared__ float data_sum;

    //Allocate arrays for the gradients etc ...
    //float dudt[NUM_VARS];
    //float d2udt2[NUM_VARS];

    __shared__ float dudt[NUM_VARS];
    __shared__ float d2udt2[NUM_VARS];

    //two temporary arrays which are used whenever we want to reduce (sum, min, max)
    //across threads
    volatile __shared__ float s_temp0[BLOCK_MAX_SIZE];
    volatile __shared__ float s_temp1[BLOCK_MAX_SIZE];
    
    //The numerator and denominator for the update step. These are used from multiple threads, and need
    //to be shared
    volatile __shared__ float NR_Numerator[NUM_VARS];
    volatile __shared__ float NR_Denominator[NUM_VARS];

    //the parameter vector
    volatile __shared__ float theta[NUM_VARS];

    //can these be const??? I don't see maxjump being revised anywhere
    const float maxjump[NUM_VARS]={1e0, 1e0, 1e2, 2e0}; // maxjump is revised such that the intensity can converge to the correct value
    const float gamma[NUM_VARS]={1.0, 1.0, 0.5, 1.0};
    
    //variable to store maximum intensity in ROI
    //float Nmax;

    //Prevent read/write past end of array
    //if ((bx*BlockSize+tx)>=Nfits) return;
    
    //zero arrays (each thread zeros one location)
    // As we have more pixels than entries in the matrix, we will only use the first NV*NV threads
    // We've effectively parallized the loop across the threads
    // CHECK - Do we still need to zero M (we now do the sum and then assign, rather than a progressive sum)
    if (pixelIndex < NUM_VARS*NUM_VARS){
        M[pixelIndex] = 0;
        Minv[pixelIndex] = 0;
    }
    

    //maybe put a __syncthreads here to ensure following ops are coallesced??
    __syncthreads(); //Is this necessary?

    //load data from global memory (each thread loads the data for it's pixel)
    //as each thread accesses a consecutive memory location, access should be coalesced
    pixel_data = d_data[blockOffset + pixelIndex];
    pixel_variance =(d_varim[blockOffset + pixelIndex]);
    pixel_gain = (d_gainim[blockOffset + pixelIndex]);
    
    //Precompute these quantities (previously re-evaluated on each iteration)
    pixel_variance_over_gain_squared = pixel_variance/(pixel_gain*pixel_gain);
    data=pixel_data + pixel_variance_over_gain_squared;

    //wait untill all loads have taken place
    __syncthreads();

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

    /*if (pixelIndex == 0){
        printf("data_sum: %f\n", data_sum);
        printf("x0, y0: %f, %f\n", theta[0], theta[1]);
    }*/

    //find max and min values
    s_temp0[pixelIndex] = pixel_data;
    s_temp1[pixelIndex] = pixel_data;

    kernel_min_reduce(BlockSize, pixelIndex, s_temp0, &theta[3]);
    kernel_max_reduce(BlockSize, pixelIndex, s_temp1, &tmp);
    
    if (pixelIndex == 0)
        //We only need to do this once - do it in thread 0
        theta[2]=max(0.0, (tmp-theta[3])*2*pi*PSFSigma*PSFSigma);

    __syncthreads();
    
    for (kk=0;kk<iterations;kk++) {//main iterative loop

        PSFx=kernel_IntGauss1D(threadIdx.x, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(threadIdx.y, theta[1], PSFSigma);
            
        model=theta[3] + theta[2]*PSFx*PSFy + pixel_variance_over_gain_squared;
        //data=pixel_data + pixel_variance_over_gain_squared; //now calculated outside loop
            
        //calculating derivatives
        kernel_DerivativeIntGauss1D(threadIdx.x, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
        kernel_DerivativeIntGauss1D(threadIdx.y, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
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
            
        for (ll=0;ll < NUM_VARS;ll++){
            s_temp0[pixelIndex] = dudt[ll]*cf;
            s_temp1[pixelIndex] = d2udt2[ll]*cf - pow(dudt[ll], 2)*df;

            //sum over all pixels
            kernel_sum_reduce_dual(BlockSize, pixelIndex, s_temp0, s_temp1, &NR_Numerator[ll], &NR_Denominator[ll]);
        }

        
        // The update
        ll = pixelIndex;
        //rather than looping over the variables, use the first NUM_VARS threads
        if (ll < NUM_VARS){
            if (kk<2)
                theta[ll]-=gamma[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
            else
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        }
        __syncthreads();
        
        if (pixelIndex == 0){
            // Do this in the 0th thread only - i.e. once per block/ROI
            // Any other constraints
            theta[2]=max(theta[2], 1.0);
            theta[3]=max(theta[3], 0.01);

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
        PSFx=kernel_IntGauss1D(threadIdx.x, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(threadIdx.y, theta[1], PSFSigma);
            
        model=theta[3] + theta[2]*PSFx*PSFy + pixel_variance_over_gain_squared;
            
        //calculating derivatives
        kernel_DerivativeIntGauss1D(threadIdx.x, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(threadIdx.y, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
        
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
            
        //Building the Fisher Information Matrix
        for (kk=0;kk<NUM_VARS;kk++) for (ll=kk;ll<NUM_VARS;ll++){

            //reuse s_temp0 here to save memory
            s_temp0[pixelIndex] = dudt[ll]*dudt[kk]/model;

            //sum over all pixels
            kernel_sum_reduce(BlockSize, pixelIndex, s_temp0, &M[kk*NUM_VARS+ll]);

            if (pixelIndex == 0){
                //execute in first thread of block only

                //exploit known matrix symetry and copy value to the mirrored entry
                M[ll*NUM_VARS+kk] = M[kk*NUM_VARS+ll];
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
        kernel_sum_reduce(BlockSize, pixelIndex, s_temp0, &tmp);

        if (pixelIndex == 0){
            //only in first thread ...

            //write our log likelihood back into global memory
            d_LogLikelihood[blockIdx.x] = tmp;
        
            // Matrix inverse (CRLB=F^-1) and output assigments 
            kernel_MatInvN(M, Minv, Diag, NUM_VARS);
        }

    }
    __syncthreads();
    
    //write to global arrays
    //CHECK!! - are output arrays transposed???
    //We want this matrix orientation as it optimizes memory access speed (coalescing) 
    if (pixelIndex < NUM_VARS){
        d_Parameters[NUM_VARS*blockIdx.x + pixelIndex] = theta[pixelIndex];
        d_CRLBs[NUM_VARS*blockIdx.x + pixelIndex] = Diag[pixelIndex];
    }
    
    return;
}

#define NUM_VARS_ASTIG 6

__global__ void kernel_MLEFit_pix_threads_astig(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim, int calcCRB){
    /* A version of MLEFit that uses per-pixel, rather than per fit threads 

    Each block consists corresponds to one ROI. threadIdx.x is the x pixel coordinate, 
    threadIdx.y is the y coordinate.

    David Baddeley
    Feb 2016

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
    int blockOffset = BlockSize*blockIdx.x; //the offset in memory to the current ROI
    int pixelIndex = threadIdx.x + blockDim.x*threadIdx.y; //index of pixel in ROI

    //printf("pixelIndex: %d\n", pixelIndex);

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
    pixel_data = d_data[blockOffset + pixelIndex];
    pixel_variance = d_varim[blockOffset + pixelIndex];
    pixel_gain = d_gainim[blockOffset + pixelIndex];
    
    //Precompute these quantities (previously re-evaluated on each iteration)
    pixel_variance_over_gain_squared = pixel_variance/(pixel_gain*pixel_gain);
    data=pixel_data + pixel_variance_over_gain_squared;

    //wait untill all loads have taken place
    __syncthreads();

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

    /*if (pixelIndex == 0){
        printf("data_sum: %f\n", data_sum);
        printf("x0, y0: %f, %f\n", theta[0], theta[1]);
    }*/

    //find max and min values
    s_temp0[pixelIndex] = pixel_data;
    s_temp1[pixelIndex] = pixel_data;

    kernel_min_reduce(BlockSize, pixelIndex, s_temp0, &theta[3]);
    kernel_max_reduce(BlockSize, pixelIndex, s_temp1, &Nmax);
    
    if (pixelIndex == 0)
        //We only need to do this once - do it in thread 0
        theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);

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
            if (kk<2)
                theta[ll]-=gamma[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
            else
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        }
        __syncthreads();
        
        if (pixelIndex == 0){
            // Do this in the 0th thread only - i.e. once per block/ROI
            // Any other constraints
            theta[2]=max(theta[2], 1.0);
            theta[3]=max(theta[3], 0.01);
            theta[4]=max(theta[4], PSFSigma/10.0);
            theta[5]=max(theta[5], PSFSigma/10.0); 

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
        kernel_DerivativeIntGauss1D(threadIdx.x, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(threadIdx.y, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
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
    
    //write to global arrays
    //CHECK!! - are output arrays transposed???
    //We want this matrix orientation as it optimizes memory access speed (coalescing) 
    if (pixelIndex < NUM_VARS_ASTIG){
        d_Parameters[NUM_VARS_ASTIG*blockIdx.x + pixelIndex] = theta[pixelIndex];
        d_CRLBs[NUM_VARS_ASTIG*blockIdx.x + pixelIndex] = Diag[pixelIndex];
    }
    
    return;
}



//theta is: {x,y,N,bg}
__global__ void kernel_MLEFit(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim){
    
    __shared__ float s_data[MEM];
	__shared__ float s_varim[MEM];
	__shared__ float s_gainim[MEM];
    float M[NV_P*NV_P], Diag[NV_P], Minv[NV_P*NV_P];
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int BlockSize = blockDim.x; //Number of threads in current block 

	//float gain=0.45;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=NV_P;
    float dudt[NV_P];
    float d2udt2[NV_P];
    float NR_Numerator[NV_P], NR_Denominator[NV_P];
    float theta[NV_P];
    float maxjump[NV_P]={1e0, 1e0, 1e2, 2e0}; // maxjump is revised such that the intensity can converge to the correct value
    float gamma[NV_P]={1.0, 1.0, 0.5, 1.0};
    float Nmax;

    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
    //load data
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++)
	{
        /*s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_varim[sz*sz*tx+sz*jj+ii]=d_varim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_gainim[sz*sz*tx+sz*jj+ii]=d_gainim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];*/

        s_data[sz*sz*tx + sz*jj + ii]=d_data[sz*sz*bx*BlockSize + sz*sz*tx + sz*jj + ii];
        s_varim[sz*sz*tx + sz*jj + ii]=d_varim[sz*sz*bx*BlockSize + sz*sz*tx + sz*jj+ii];
        s_gainim[sz*sz*tx + sz*jj + ii]=d_gainim[sz*sz*bx*BlockSize + sz*sz*tx + sz*jj+ii];
	}
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
            PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);
            
            model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            
            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], &d2udt2[1]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
        if (kk<2)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=gamma[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        
        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);
        
        model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        
        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk + BlockSize*bx + tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk + BlockSize*bx + tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx + tx] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;
    
    return;
}

__global__ void kernel_MLEFit_sigma(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim, float *d_gainim){
    
    __shared__ float s_data[MEM];
	__shared__ float s_varim[MEM];
	__shared__ float s_gainim[MEM];
    float M[NV_PS*NV_PS], Diag[NV_PS], Minv[NV_PS*NV_PS];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
	//float gain=0.45;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=NV_PS;
    float dudt[NV_PS];
    float d2udt2[NV_PS];
    float NR_Numerator[NV_PS], NR_Denominator[NV_PS];
    float theta[NV_PS];
    float maxjump[NV_PS]={1e0, 1e0, 1e2, 2e0, 5e-1};
    float gamma[NV_PS]={1.0, 1.0, 0.5, 1.0, 1.0};
    float Nmax;
    
    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
      
    //copy in data
	for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_varim[sz*sz*tx+sz*jj+ii]=d_varim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
		s_gainim[sz*sz*tx+sz*jj+ii]=d_gainim[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
	}
    
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;
    
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
            PSFy=kernel_IntGauss1D(jj, theta[1], theta[4]);
            
            model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
            
            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], &d2udt2[4]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
        if (kk<5)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=gamma[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        
        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        theta[4]=max(theta[4], 0.5);
        theta[4]=min(theta[4], sz/2.0);
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
        PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);
        
        model=theta[3]+theta[2]*PSFx*PSFy+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        data=s_data[sz*sz*tx+sz*jj+ii]+s_varim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii]/s_gainim[sz*sz*tx+sz*jj+ii];
        
        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss2DSigma(ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
  
    
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
	//d_LogLikelihood[BlockSize*bx+tx] = 1;
    
    
    return;
}

//******************* 3D localization using z fit does not work for this version of sCMOS software **********************//
__global__ void kernel_MLEFit_z(float *d_data, float PSFSigma_x, float Ax, float Ay, float Bx, float By, float gamma, float d, float PSFSigma_y, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim){
    __shared__ float s_data[MEM];
    float M[5*5], Diag[5], Minv[5*5];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=5;
    float dudt[5];
    float d2udt2[5];
    float NR_Numerator[5], NR_Denominator[5];
    float theta[5];
    float maxjump[5]={1e0, 1e0, 1e2, 2e0, 1e-1};
    float g[5]={1.0, 1.0, 0.5, 1.0, 1.0};
    float Nmax;
    
    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;

    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
    
    //copy in data
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++)
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
    
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma_x, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma_x*PSFSigma_y*sqrt(2.0f));
    theta[4]=0;
   
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay,Bx,By, gamma, d, &PSFx, &PSFy, dudt, d2udt2);
            
            model=theta[3]+theta[2]*PSFx*PSFy;
            data=s_data[sz*sz*tx+sz*jj+ii];
            
            //calculating remaining derivatives
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
        if (kk<2)
            for (ll=0;ll<NV;ll++)
                theta[ll]-=g[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        else
            for (ll=0;ll<NV;ll++)
                theta[ll]-=min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);
        
        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        
        kernel_DerivativeIntGauss2Dz(ii, jj, theta, PSFSigma_x,PSFSigma_y, Ax,Ay, Bx,By, gamma, d, &PSFx, &PSFy, dudt, NULL);
        
        model=theta[3]+theta[2]*PSFx*PSFy;
        data=s_data[sz*sz*tx+sz*jj+ii];
        
        //calculating remaining derivatives
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
       
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) 
    kernel_MatInvN(M, Minv, Diag, NV);
  
   //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
    return;
}

//************* sigma fit does not work for this version of sCMOS software***********************************//
__global__ void kernel_MLEFit_sigmaxy(float *d_data, float PSFSigma, int sz, int iterations, 
        float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,int Nfits, float *d_varim){
 
    __shared__ float s_data[MEM]; //note doesn't allocate memory for gain or variance
    float M[6*6], Diag[6], Minv[6*6];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int BlockSize = blockDim.x;
    int ii, jj, kk, ll;
    float model, cf, df, data;
    float Div;
    float PSFy, PSFx;
    int NV=6;
    float dudt[6];
    float d2udt2[6];
    float NR_Numerator[6], NR_Denominator[6];
    float theta[6];
    float maxjump[6]={1e0, 1e0, 1e2, 2e0, 1e-1,1e-1};
    float g[6]={1.0, 1.0, 0.5, 1.0, 1.0,1.0};
    float Nmax;
    
    //Prevent read/write past end of array
    if ((bx*BlockSize+tx)>=Nfits) return;
    
    for (ii=0;ii<NV*NV;ii++)M[ii]=0;
    for (ii=0;ii<NV*NV;ii++)Minv[ii]=0;
    
    //copy in data
    
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++)
        s_data[sz*sz*tx+sz*jj+ii]=d_data[sz*sz*bx*BlockSize+sz*sz*tx+sz*jj+ii];
    
    //initial values
    kernel_CenterofMass2D(sz, &s_data[sz*sz*tx], &theta[0], &theta[1]);
    kernel_GaussFMaxMin2D(sz, PSFSigma, &s_data[sz*sz*tx], &Nmax, &theta[3]);
    theta[2]=max(0.0, (Nmax-theta[3])*2*pi*PSFSigma*PSFSigma);
    theta[4]=PSFSigma;
    theta[5]=PSFSigma;
    for (kk=0;kk<iterations;kk++) {//main iterative loop
        
        //initialize
        for (ll=0;ll<NV;ll++){
            NR_Numerator[ll]=0;
            NR_Denominator[ll]=0;}
        
        for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
            PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
            PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
            
            model=theta[3]+theta[2]*PSFx*PSFy;
            data=s_data[sz*sz*tx+sz*jj+ii];
            
            //calculating derivatives
            kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], &d2udt2[0]);
            kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], &d2udt2[1]);
            kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], &d2udt2[4]);
            kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], &d2udt2[5]);
            dudt[2] = PSFx*PSFy;
            d2udt2[2] = 0.0;
            dudt[3] = 1.0;
            d2udt2[3] = 0.0;
            
            cf=0.0;
            df=0.0;
            if (model>10e-3) cf=data/model-1;
            if (model>10e-3) df=data/pow(model, 2);
            cf=min(cf, 10e4);
            df=min(df, 10e4);
            
            for (ll=0;ll<NV;ll++){
                NR_Numerator[ll]+=dudt[ll]*cf;
                NR_Denominator[ll]+=d2udt2[ll]*cf-pow(dudt[ll], 2)*df;
            }
        }
        
        // The update
            for (ll=0;ll<NV;ll++)
                theta[ll]-=g[ll]*min(max(NR_Numerator[ll]/NR_Denominator[ll], -maxjump[ll]), maxjump[ll]);

        // Any other constraints
        theta[2]=max(theta[2], 1.0);
        theta[3]=max(theta[3], 0.01);
        theta[4]=max(theta[4], PSFSigma/10.0);
        theta[5]=max(theta[5], PSFSigma/10.0);  
    }
    
    // Calculating the CRLB and LogLikelihood
    Div=0.0;
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        
        PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
        PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
        
        model=theta[3]+theta[2]*PSFx*PSFy;
        data=s_data[sz*sz*tx+sz*jj+ii];
        
        //calculating derivatives
        kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
        kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
        kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
        kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);
        dudt[2] = PSFx*PSFy;
        dudt[3] = 1.0;
        
        //Building the Fisher Information Matrix
        for (kk=0;kk<NV;kk++)for (ll=kk;ll<NV;ll++){
            M[kk*NV+ll]+= dudt[ll]*dudt[kk]/model;
            M[ll*NV+kk]=M[kk*NV+ll];
        }
        
        //LogLikelyhood
        if (model>0)
            if (data>0)Div+=data*log(model)-model-data*log(data)+data;
            else
                Div+=-model;
    }
    
    // Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV);
   
    //write to global arrays
    for (kk=0;kk<NV;kk++) d_Parameters[Nfits*kk+BlockSize*bx+tx]=theta[kk];
    for (kk=0;kk<NV;kk++) d_CRLBs[Nfits*kk+BlockSize*bx+tx]=Diag[kk];
    d_LogLikelihood[BlockSize*bx+tx] = Div;
    return;
}
