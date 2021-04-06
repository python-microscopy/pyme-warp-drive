/*
findPeaks.cu contains kernels to perform maximum filter, and from that extract candidate molecule positions

Andrew Barentine - Spring 2016
*/

#include <math.h> // for fmaxf

__global__ void maxfRowGPU(float *data, float *rconvdata, 
                           const int half_filter_size)
/* 
This function takes input data and performs a row max filter. The 
convolution is stored in a separate output array.

CUDA indexing
-------------
block
    x: n_columns
        size[1] of data
grid
    x: n_rows
        size[0] of data

Each row is loaded into shared memory before the convolution is performed. 
Currently, the maximum size array that can be convolved by this function is 
1024x1024, because each pixel is assigned its own thread.
*/
{
    int k;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_max = 0;

    volatile __shared__ float rdata_sh[1075]; //FIXME: should be changed to colsize (PADDED SIZE, or larger)
    
    // Pad the shared memory array
    if (threadIdx.x < half_filter_size){
        rdata_sh[threadIdx.x] = 0;
        rdata_sh[blockDim.x + threadIdx.x + half_filter_size] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    rdata_sh[threadIdx.x + half_filter_size] = data[ind];

    __syncthreads();  // make sure we are ready for the row convolution

    for (k = -half_filter_size; k <= half_filter_size - 1; k++){
        temp_max = fmaxf(rdata_sh[(threadIdx.x + half_filter_size) - k], temp_max);
    }
    rconvdata[ind] = temp_max;

}
__global__ void maxfColGPU(float *rconvdata, const int half_filter_size)
/* 
This function takes input data and performs a column max filter. The 
convolution is stored in a separate output array.

CUDA indexing
-------------
block
    x: n_rows
        size[0] of data
grid
    x: n_columns
        size[1] of data

Each column is loaded into shared memory before the convolution is performed. 
Currently, the maximum size array that can be convolved by this function is 
1024x1024, because each pixel is assigned its own thread.
*/
{
    int k;
    int ind = blockIdx.x + gridDim.x * threadIdx.x;
    float temp_max = 0;

    volatile __shared__ float cdata_sh[1075]; //FIXME: should be changed to colsize (PADDED SIZE, or larger)

    if (threadIdx.x < half_filter_size){
        cdata_sh[threadIdx.x] = 0;
        cdata_sh[blockDim.x + threadIdx.x + half_filter_size] = 0;
    }
    cdata_sh[threadIdx.x + half_filter_size] = rconvdata[ind];

    __syncthreads();

    for (k = -half_filter_size; k <= half_filter_size - 1; k++){
        tem_pmax = fmaxf(cdata_sh[(threadIdx.x + half_filter_size) - k], temp_max);
    }
    rconvdata[ind] = temp_max;

}

__global__ void findPeaks(float *unif, float *maxfData, float thresh, const int colsize, int *counter, int *candPos,
const int halfROIsize, const int maxCandCount, float *noiseSig, const float electronsPerADU){
/*


args:
    unif:
    maxfData:
    thresh:
    colsize:
    counter:
    candPos:
    halfROIsize:
    maxCandCount:
    noiseSig: a dummy variable existing solely to create symmetry between this call and that of findPeaksSNThresh
    electronsPerADU: a dummy variable existing solely to create symmetry between this call and that of findPeaksSNThresh


*/
int dloc = blockIdx.x*colsize + threadIdx.x;
int temp;

//maxfData[dloc] = (unif[dloc] >= 0.99999999*maxfData[dloc]) && (unif[dloc] > thresh);

//if ((unif[dloc] >= 0.99999999*maxfData[dloc]) && (unif[dloc] > thresh)){
if ((unif[dloc] == maxfData[dloc]) && (unif[dloc] > thresh)){
    if ((blockIdx.x > (halfROIsize + 2)) && ((gridDim.x - blockIdx.x) > (halfROIsize + 2)) && (threadIdx.x > (halfROIsize + 2)) &&((colsize - threadIdx.x) > (halfROIsize + 2))){
        maxfData[dloc] = 1;
        temp = atomicAdd(counter, 1);  // returns old, i.e. we're still 0-indexed
        if (*counter < maxCandCount){
            candPos[temp] = dloc;
        }

    }

}
    else {
        maxfData[dloc] = 0;
    }


}

__global__ void find_candidates_noise_thresh(float *dog_data, float *maxf_data, float thresh_factor, int *counter,
int *candidate_indices, const int max_cand_count, const int half_roi_size, float *noise_sigma)
/*

    Applies a pixel-specific noise-based threshold to smoothed input data to find candidate molecules.

    dog_data: difference of gaussian (dog) filtered data. [e-]
    maxf_data: max-filtered copy of dog_data. [e-]
    thresh_factor: multiple of noise sigma to reject molecules below
    counter: number of molecules which have been found, used as index for writing out candidate position array. Access
        should always be atomic.
    candidate_indices: memory allocation to write-out candidate molecule indices
    max_cand_count: length of candidate_indices array, so we can check that we don't try and write past it. Maximum
        number of candidates that can be detected on a single frame.
    half_roi_size: half size of the ROI [pixels]
    noise_sigma: per-pixel noise standard deviation [e-]

    CUDA indexing
    -------------
    block
        x: n_columns
            size[1] of the data frame
    grid
        x: n_rows
            size[0] of the data frame
*/
{
    int dloc = blockIdx.x * blockDim.x + threadIdx.x;
    int temp_ind;

    // rather than pycuda memcpy a zero before calling this kernel, zero it here each time
    if (dloc == 0){
        *counter = 0;
    }
    __syncthreads();  // make sure it's zero before any threads start using it

    // multiply threshold factor by the noise standard deviation
    float threshold = thresh_factor * noise_sigma[dloc];


    if ((dog_data[dloc] == maxf_data[dloc]) && (dog_data[dloc] > threshold)){
        if ((blockIdx.x > (half_roi_size + 2)) && ((gridDim.x - blockIdx.x) > (half_roi_size + 2))
           && (threadIdx.x > (half_roi_size + 2)) && ((gridDim.x - threadIdx.x) > (half_roi_size + 2))){
            //maxf_data[dloc] = 1;
            temp_ind = atomicAdd(counter, 1);  // returns old, i.e. we're still 0-indexed
            if (*counter < max_cand_count){
                candidate_indices[temp_ind] = dloc;
            }
        }
    }
    //else {
    //    maxf_data[dloc] = 0;
    //}
}

/*
__global__ void getROIs(float *data, float *roiStack, int *candPos, const int colsize, const int rowsize,
const int halfROIsize){

int cand = blockIdx.x;
int jj = threadIdx.x;
int kk = threadIdx.y;

int ulcROI = candPos[cand] - halfROIsize - halfROIsize*colsize; //Element of the upper left hand corner of the ROI

float temp1 = fmodf(int2float(candPos[cand]), int2float(colsize)) ;
float temp2 = fmodf(int2float(candPos[cand])/colsize, int2float(rowsize));

//if ((temp > halfROIsize + 1) && temp < colsize - halfROIsize){
if ((temp1 > (halfROIsize + 1)) && (temp2 < (halfROIsize + 1))){
    roiStack[jj + (2*halfROIsize)*kk + 4*halfROIsize*cand] = data[candPos[cand] + jj + kk*colsize];
}

}*/