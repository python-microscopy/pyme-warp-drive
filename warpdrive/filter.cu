/*
filter.cu contains various convolution functions for the detector class to call.

Andrew Barentine - Spring 2016
*/

#include <stdio.h>

__global__ void dog_row_variance_convolution(float *var, float *rconvdata, float *filter,// const int rowsize,
int halfFilt, const int colsize)
/*
This function takes input data and performs a row convolution. The convolution is stored in a separate
output array, and is performed on the inverse of the incoming data (variance).

Each row is loaded into shared memory before the convolution is performed. Currently, the maximum size array that can
be convolved by this function is 1024x1024, because each pixel is assigned its own thread.
*/
{
    int k, halfFiltm1 = halfFilt-1;
    int rid = blockIdx.x;// + halfFilt;
    int j = threadIdx.x;// + halfFilt;
    float tempsum = 0;

    volatile __shared__ float rdata_sh[1075]; //FIXME: should be changed to colsize (PADDED SIZE, or larger)
    __shared__ float filter_sh[12];

    // pad row of shared memory
    if (j < (halfFilt)){
        rdata_sh[j] = 0;
        rdata_sh[colsize + j + halfFilt] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    if ((j == (halfFilt-1)) && (rid == 1)){
        printf("For Gondor!!");
    }
    // load row of data into shared mem and weight on variance
    rdata_sh[j + halfFilt] = 1/var[rid*colsize + j];

    // Load filter into shared memory
    if (j < (2*halfFilt)) filter_sh[j] = filter[j];

    // make sure we've loaded everything we need to
    __syncthreads();

    //if ((j==colsize-1)&&(rid==5)) printf("colsize + halfFilt %d", (colsize + j));

    // Perform convolution
    for (k = -halfFilt; k <= halfFiltm1; k++){
        tempsum += rdata_sh[(j + halfFilt) - k]*filter_sh[k + halfFilt];
    }
    // push to output array
    rconvdata[rid*colsize + j] = tempsum;

}


__global__ void convRowGPU_comb(float *data, float *var, float *rowConvBig, float *rowConvSmall,
float *filterBig, float *filterSmall, int halfFiltBig, int halfFiltSmall, int colsize)
/*
THIS FUNCTION HAS BEEN DEPRECIATED, as for some reason the maximum array size it can convolve is less than 1024x1024
This function takes input data and performs TWO row convolutions. The convolutions are stored in separate
output arrays.

Each row is loaded into shared memory before the convolution is performed.
*/
{
    int k;
    int rid = blockIdx.x;// + threadIdx.y;
    int j = threadIdx.x;
    float tempsum = 0;

    //printf("threadIdx.y %d", threadIdx.y);

    volatile __shared__ float rdata_sh[1075]; //must be padded-image size + 1, or larger)
    //volatile __shared__ float rdata2_sh[275];
    __shared__ float filterBig_sh[12]; //must be at least the filter size
    __shared__ float filterSmall_sh[12]; //must be at least the filter size

    if (j < (halfFiltBig)){
        rdata_sh[j] = 0;
        rdata_sh[colsize + j + halfFiltBig] = 0;
        //printf("halfFiltBig, halfFiltSmall %d", halfFiltSmall);
    }
    rdata_sh[j + halfFiltBig] = data[rid*colsize + j]/var[rid*colsize + j];
    // Note that in the future, these next two lines could be combined by sending
    // in the filters concatenated into a single array
    if (j < (2*halfFiltBig)){
        filterBig_sh[j] = filterBig[j];
    }
    if (j < (2*halfFiltSmall)){
        filterSmall_sh[j] = filterSmall[j];
    }

    __syncthreads(); //make sure each thread is finished writing to shared memory

    //printf("threadIdx.y %d", threadIdx.y);
    for (k = -halfFiltBig; k < halfFiltBig; k++){
        tempsum += rdata_sh[(j + halfFiltBig) - k]*filterBig_sh[k + halfFiltBig];
    }
    rowConvBig[rid*colsize + j] = tempsum;
    //rowConvBig[rid*colsize + j] = rdata_sh[j + halfFiltBig][threadIdx.y];

    tempsum = 0;

    for (k = -halfFiltSmall; k <= halfFiltSmall; k++){
        tempsum += rdata_sh[(j + halfFiltBig) - k]*filterSmall_sh[k + halfFiltSmall];
    }
    rowConvSmall[rid*colsize + j] = tempsum;
    //rowConvSmall[rid*colsize + j] = rdata_sh[j + halfFiltBig][threadIdx.y];

}

__global__ void dog_row_convolution(float *data, float *var, float *row_convolved_data, float *filter,// const int rowsize,
int half_filter_size, float *background)
/*
    Perform the first part of a separable convolution. FIXME - finish this description

    Parameters
    ----------
    data: input data, camera-corrected and converted to units of e-
    var: (per-pixel) variance due to readout noise [e-^2]
    rconvdata: memory allocation to store result


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
/*
This function takes input data, subtracts the pixel-dependent background estimate, converts the data from units of ADU
to photoelectrons and performs a row convolution. The convolution is stored in a separate output array.
Each row is loaded into shared memory before the convolution is performed. Currently, the maximum size array that can
be convolved by this function is 1024x1024, because each pixel is assigned its own thread.
*/
{
    int k;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_sum = 0;

    volatile __shared__ float rdata_sh[1075]; //should be changed to blockDim.x (PADDED SIZE, or larger)
    __shared__ float filter_sh[12];

    // Pad the shared memory array
    if (threadIdx.x < (half_filter_size)){
        rdata_sh[threadIdx.x] = 0;
        rdata_sh[blockDim.x + threadIdx.x + half_filter_size] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    // load row of data into shared mem and subtract background
    rdata_sh[threadIdx.x + half_filter_size] = (data[ind] - background[ind])/var[ind];
    if (threadIdx.x < (2 * half_filter_size)) filter_sh[threadIdx.x] = filter[threadIdx.x];

    // make sure we're ready to convolve
    __syncthreads();

    // perform convolution
    for (k = -half_filter_size; k <= half_filter_size - 1; k++){
        temp_sum += rdata_sh[(threadIdx.x + half_filter_size) - k]*filter_sh[k + half_filter_size];
    }
    // push results to output array
    row_convolved_data[ind] = temp_sum;
}


__global__ void dog_column_convolution(float *data,  float *filter, int rowsize, int colsize, int halfFilt)
{
/*
This function takes input data (row convolved data) and performs a column convolution. The convolution results are
stored in the original input array.

Each row is loaded into shared memory before the convolution is performed. Currently, the maximum size array that can
be convolved by this function is 1024x1024, because each pixel is assigned its own thread.
*/
    int k, halfFiltm1 = halfFilt-1;
    int cid = blockIdx.x;// + halfFilt;
    int j = threadIdx.x;// + halfFilt;
    float tempsum = 0;

    // allocated column of shared memory
    volatile __shared__ float cdata_sh[1075]; //should be changed to rowsize
    __shared__ float filter_sh[12];

    // pad shared mem column
    if (j < (halfFilt)){
        cdata_sh[j] = 0;
        cdata_sh[rowsize + j + halfFilt] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    // load data column into shared mem
    cdata_sh[j + halfFilt] = data[j*colsize + cid];
    // load filter into shared mem
    if (j < (2*halfFilt)) filter_sh[j] = filter[j];

    // make sure we are ready for the convolution
    __syncthreads();

    // perform convolution
    for (k = -halfFilt; k <= halfFiltm1; k++){
        tempsum += cdata_sh[(j + halfFilt) - k]*filter_sh[k + halfFilt];
    }

    // store results in input array
    data[cid + j*colsize] = tempsum;
}

__global__ void weighted_difference_of_gaussian_subtraction(float *uniflargedat, float *uniflargevar, float *unifsmalldat,  float *unifsmallvar,
int colsize, int halfFilt)
{
/*
Perform the necessary divisions and subtractions on previously convolved arrays to achieve the filtering described in
supplemental materials of 10.1038/nmeth.2488.
*/
    int rid = blockIdx.x;
    int cid = threadIdx.x;
    int dloc = rid * colsize + cid;

    uniflargedat[dloc] = (unifsmalldat[dloc]/unifsmallvar[dloc]) - (uniflargedat[dloc]/uniflargevar[dloc]);

}
