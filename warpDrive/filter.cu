#include <stdio.h>

__global__ void convRowGPU_var(float *var, float *rconvdata, float *filter,// const int rowsize,
int halfFilt, const int colsize)
//
// This function takes input data and performs a row convolution. The convolution is stored in a separate
// output array.
//
{
    int k, halfFiltm1 = halfFilt-1;
    int rid = blockIdx.x;// + halfFilt;
    int j = threadIdx.x;// + halfFilt;
    float tempsum = 0;

    volatile __shared__ float rdata_sh[1075]; //should be changed to colsize (PADDED SIZE, or larger)
    __shared__ float filter_sh[12];
    //rdata_sh[j] = data[rid*colsize + j];


    if (j < (halfFilt)){
        rdata_sh[j] = 0;
        rdata_sh[colsize + j + halfFilt] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    rdata_sh[j + halfFilt] = 1/var[rid*colsize + j];
    if (j < (2*halfFilt)) filter_sh[j] = filter[j];

    //if (j > (colsize - 2*halfFilt)){
    //    filter_sh[j] = filter[j]; //trying to use other threads in case the two threads can run async.
    //    printf("%d,", j);
    //}

    __syncthreads();

    //if ((j==colsize-1)&&(rid==5)) printf("colsize + halfFilt %d", (colsize + j));

    for (k = -halfFilt; k <= halfFiltm1; k++){
        tempsum += rdata_sh[(j + halfFilt) - k]*filter_sh[k + halfFilt];
    }
    rconvdata[rid*colsize + j] = tempsum;
    //rconvdata[rid*colsize + j] = rdata_sh[j + halfFilt];
}


__global__ void convRowGPU_comb(float *data, float *var, float *rowConvBig, float *rowConvSmall,
float *filterBig, float *filterSmall, int halfFiltBig, int halfFiltSmall, int colsize)
/*
convRowGPU
*/
{
    int k;
    int rid = blockIdx.x;// + threadIdx.y;
    int j = threadIdx.x;
    float tempsum = 0;

    //printf("threadIdx.y %d", threadIdx.y);

    volatile __shared__ float rdata_sh[275]; //must be padded image size + 1, or larger)
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

__global__ void convRowGPU(float *data, float *var, float *rconvdata, float *gain, float *filter,// const int rowsize,
int halfFilt, const int colsize)
//
// This function takes input data and performs a row convolution. The convolution is stored in a separate
// output array.
//
{
    int k, halfFiltm1 = halfFilt-1;
    int rid = blockIdx.x;// + halfFilt;
    int j = threadIdx.x;// + halfFilt;
    float tempsum = 0;

    volatile __shared__ float rdata_sh[1075]; //should be changed to colsize (PADDED SIZE, or larger)
    __shared__ float filter_sh[12];
    //rdata_sh[j] = data[rid*colsize + j];


    if (j < (halfFilt)){
        rdata_sh[j] = 0;
        rdata_sh[colsize + j + halfFilt] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    rdata_sh[j + halfFilt] = data[rid*colsize + j]/(var[rid*colsize + j]*gain[rid*colsize + j]);
    if (j < (2*halfFilt)) filter_sh[j] = filter[j];

    //if (j > (colsize - 2*halfFilt)){
    //    filter_sh[j] = filter[j]; //trying to use other threads in case the two threads can run async.
    //    printf("%d,", j);
    //}

    __syncthreads();

    //if ((j==colsize-1)&&(rid==5)) printf("colsize + halfFilt %d", (colsize + j));

    for (k = -halfFilt; k <= halfFiltm1; k++){
        tempsum += rdata_sh[(j + halfFilt) - k]*filter_sh[k + halfFilt];
    }
    rconvdata[rid*colsize + j] = tempsum;
    //rconvdata[rid*colsize + j] = rdata_sh[j + halfFilt];
}


__global__ void convColGPU(float *data,  float *filter, int rowsize, int colsize, int halfFilt)
{
//
// convColGPU takes rov-convoled data as an input and performs a column-wise convolution. The
// completed 2D convolution is then stored in a separate output array.
//
    int k, halfFiltm1 = halfFilt-1;
    int cid = blockIdx.x;// + halfFilt;
    int j = threadIdx.x;// + halfFilt;
    float tempsum = 0;


    volatile __shared__ float cdata_sh[1075][1]; //should be changed to rowsize
    __shared__ float filter_sh[12];


    if (j < (halfFilt)){
        cdata_sh[j][0] = 0;
        cdata_sh[rowsize + j + halfFilt][0] = 0;
        //printf("colsize + halfFilt %d", (colsize + halfFilt));
    }
    cdata_sh[j + halfFilt][0] = data[j*colsize + cid];
    if (j < (2*halfFilt)) filter_sh[j] = filter[j];

    __syncthreads();


    for (k = -halfFilt; k <= halfFiltm1; k++){
        tempsum += cdata_sh[(j + halfFilt) - k][0]*filter_sh[k + halfFilt];
    }

    data[cid + j*colsize] = tempsum;
}

__global__ void smoothImGPU(float *uniflargedat, float *uniflargevar, float *unifsmalldat,  float *unifsmallvar,
int colsize, int halfFilt)
{
    int rid = blockIdx.x;
    int cid = threadIdx.x;
    int dloc = rid * colsize + cid;

    uniflargedat[dloc] = (unifsmalldat[dloc]/unifsmallvar[dloc]) - (uniflargedat[dloc]/uniflargevar[dloc]);

}


//__global__ void peakLocGPU(float *data, float *rconvdata, float *filter, const int rowsize,
//const int colsize, int halfFilt)