#include <thrust/sort.h>


extern "C" { // extern C required for PyCUDA to find our functions

__device__ int partition(float *array, int p, int r)
/*
Based on Introduction to Algorithms, 3rd edition, ISBN: 0-262-03384-4
*/
{
    int ii, jj;
    float temp, x;
    x = array[r];
    ii = p - 1;
    for (jj=p; jj<=r-1; jj++){
        if (array[jj]<= x){
            ii += 1;
            temp = array[jj];
            array[jj] = array[ii];
            array[ii] = temp;
        }
    }

    if (array[r] < array[ii + 1]){
        temp = array[ii + 1];
        array[ii + 1] = array[r];
        array[r] = temp;
    }

    return (ii + 1);
}

__device__ void quicksort(float *array, int p, int r)
/*
Based on Introduction to Algorithms, 3rd edition, ISBN: 0-262-03384-4
*/
{
    if (p < r){
        int pivot = partition(array, p, r);
        quicksort(array, p, pivot-1);
        quicksort(array, pivot + 1, r);
    }
}

__global__ void nth_value_by_pixel(float *frames, const int n, float *nth_values)
/*
    Maximum of 1024 threads per block, so call with block=(image.shape[0], 1, 1), grid=(image.shape[1], 1)
    Outdated: To be called with block dim: (image.shape[0], image.shape[1], 1)
*/
{
    int data_loc;
    float to_sort[30];

    for (int ind = 0; ind < 30; ind++){
        // row-major 3D index the same pixel in depth (fastest changing)
        //data_loc = ind + 30 * (threadIdx.y + blockDim.y * threadIdx.x);
        data_loc = ind + 30 * (blockIdx.x + gridDim.x * threadIdx.x);

        //frames[data_loc] = new_frame[threadIdx.y + threadIdx.x * blockDim.y];
        //frames[data_loc] = new_frame[blockIdx.x + threadIdx.x * gridDim.x];

        to_sort[ind] = frames[data_loc];
    }

    //thrust::sort(thrust::device, to_sort, to_sort + 30);
    quicksort(to_sort, 0, 29);

    //printf("test_val is %f", to_sort[n]);

    // should not need to sync threads
    //nth_values[threadIdx.y + threadIdx.x * blockDim.y] = to_sort[n];
    nth_values[blockIdx.x + threadIdx.x * gridDim.x] = to_sort[n];
}

 __global__ void nth_value_by_pixel_shared_quicksort(float *frames, const int n, float *nth_values)
/*
    To be executed with one warp per block (32 threads)
    block=(32, 1, 1), grid=(warp_count_x, slice_shape[1])
*/
{
    int data_loc;
    int shared_offset = 30 * threadIdx.x;
    int dim0_pixel = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float to_sort[];  // will be 32 * buffer length (i.e. blockDim.x * buffer_length)

    //FIXME - need an if statement in case slice_shape[0] % 32 != 0


    //if ((threadIdx.x == 5) & (blockIdx.x == 4) & (blockIdx.y == 3)){
    //    printf("pixel0 is %d", dim0_pixel);
    //    printf("/n data_loc0 is %d", 30*(blockIdx.y + gridDim.y * dim0_pixel));
    //}

    for (int ind = 0; ind < 30; ind++){
        // row-major 3D index the same pixel in depth (fastest changing)
        // data_loc = ind3 + dim3 * (ind2 + dim2 * ind1)
        // data_loc = ind + 30 * (blockIdx.x + gridDim.x * threadIdx.x);
        data_loc = ind + 30 * (blockIdx.y + gridDim.y * dim0_pixel);

        // recall that to_sort is a per-block variable
        to_sort[shared_offset + ind] = frames[data_loc];
    }
    // recall that to_sort is a per-block variable
    quicksort(to_sort, shared_offset, shared_offset + 29);

    //printf("test_val is %f", to_sort[n]);

    // should not need to sync threads

    // nth_values is flattened 2D
    // nth_values[blockIdx.x + threadIdx.x * gridDim.x] = to_sort[n];
    nth_values[blockIdx.y + gridDim.y * dim0_pixel] = to_sort[shared_offset + n];
}

__global__ void clear_frames(float *frames, int *frame_nums)
/*
    To be called with block dim: (image.shape[0], image.shape[1], len(frame_nums)
*/
{
    int data_loc;

    // find pixel for this thread to wipe, use threadIdx.z to clear multiple frames "at once"
    data_loc = frame_nums[threadIdx.z] + 30 * (threadIdx.y + blockDim.y * threadIdx.x);

    // set value to inf
    frames[data_loc] = 1.0f/0.0f;
}

__global__ void update_frame(float *frames, float *new_frame, const int frame_num)
/*
    Maximum of 1024 threads per block, so call with block=(image.shape[0], 1, 1), grid=(image.shape[1], 1)
    Outdated: To be called with block dim: (image.shape[0], image.shape[1], 1)
*/
{
    int data_loc;

    // row-major 3D index the same pixel in depth (fastest changing)
    //data_loc = frame_num + 30 * (threadIdx.y + blockDim.y * threadIdx.x);
    data_loc = frame_num + 30 * (blockIdx.x + gridDim.x * threadIdx.x);

    // replace value with new frame value
    //frames[data_loc] = new_frame[threadIdx.y + threadIdx.x * blockDim.y];
    frames[data_loc] = new_frame[blockIdx.x + threadIdx.x * gridDim.x];
}

//block=(32, 1, 1), grid=(warp_count_x, self.slice_shape[1])
 __global__ void subtract_b_from_a(float *a, float *b)
/*
    To be executed with one warp per block (32 threads)
    block=(32, 1, 1), grid=(warp_count_x, slice_shape[1])
*/
{
    int data_loc;
    int dim0_pixel = blockDim.x * blockIdx.x + threadIdx.x;

    a[blockIdx.y + gridDim.y * dim0_pixel] -= b[blockIdx.y + gridDim.y * dim0_pixel];
}






}