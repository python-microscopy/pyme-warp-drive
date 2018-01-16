/* Andrew Barentine, andrew.barentine@yale.edu */

#include <thrust/sort.h>


extern "C" { // extern C required for PyCUDA to find our functions if we use headers like thrust

__global__ void clear_frames(float *frames, int *frame_nums)
/*
    To be called with block dim: (image.shape[0], image.shape[1], len(frame_nums)
*/
{
    int data_loc;

    // find pixel for this thread to wipe, use threadIdx.z to clear multiple frames "at once"
    data_loc = frame_nums[threadIdx.z] + 32 * (threadIdx.y + blockDim.y * threadIdx.x);

    // set value to inf
    frames[data_loc] = 1.0f/0.0f;
}

__global__ void update_frame(float *frames, float *new_frame, const int frame_num, const int buffer_length)
/*
    Maximum of 1024 threads per block, so call with block=(image.shape[0], 1, 1), grid=(image.shape[1], 1)
    Outdated: To be called with block dim: (image.shape[0], image.shape[1], 1)
*/
{
    int data_loc;

    // row-major 3D index the same pixel in depth (fastest changing)
    //data_loc = frame_num + 30 * (threadIdx.y + blockDim.y * threadIdx.x);
    data_loc = frame_num + buffer_length * (blockIdx.x + gridDim.x * threadIdx.x);

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
    //    int data_loc;
    int dim0_pixel = blockDim.x * blockIdx.x + threadIdx.x;

    a[blockIdx.y + gridDim.y * dim0_pixel] -= b[blockIdx.y + gridDim.y * dim0_pixel];
}


__global__ void nth_value_by_pixel_search_sort(float *frames, const int n, float *nth_values)
/*
- give each pixel a warp of 32 threads. Consider making the background buffer 32 frames long for simplicity.
- have each thread pull one value into shared to_sort array.
- synchronize threads
- have each thread check sum the number of values less than the one it's responsible for
- have dummy array of length 32 filled with zeros.
- atomically add 1 to the index the thread thinks it goes to
- add return value from atomic add (0 if no other thread has written, or 1, 2, so on if there are degenerate numbers
    being sorted) to index the thread thinks it sends its number to
- store number in this (potentially) modified index

For most modern GPUs, can run 64 warps per multiprocessor. We're only sorting 64 pixels per MP at a time, but we should
do it quite fast, and we won't exceed maximum shared memory per multiprocessor (32 kB / 96 kB), so we should hit full
occupancy, with the only slow bit being the atomic adds to the dummy array. Note that if we only put 32 threads in each
block, we'll cap out on the number of blocks per multiprocessor and limit our occupancy to 50%, therefore need to
operate on two pixels per block (annoying, but not too awkward). If your data has an odd number of pixels along the 0th
direction: you're on your own.

This function should therefore be called with block=(buffer_length, 2), grid=(slice_size[0] / 2, slice_size[1]).
this means that pix_r = 2* blockIdx.x + threadIdx.y; r_size = gridDim.x * 2; c_size = gridDim.y; and pix_c = blockIdx.y;
Also, the depth dimension of frames is given by buffer_length, e.g. 32, and the index is
threadIdx.x + buffer_size * threadIdx.y;
NOTE - in the interest of performance, we'll fix buffer length to 32 pixels (dynamically allocating memory may hurt speed)
and can write another version of this function to be dynamic if needed.
*/
{

    int shared_offset = 32 * threadIdx.y; // offset since we're doing two pixels per block
    //extern int degeneracy_counter[];
    //extern __shared__ float to_sort[];  // will be 2 * buffer length (e.g. 64)
    __shared__ int degeneracy_counter[64];
    __shared__ float to_sort[64];  // will be 2 * buffer length (e.g. 64)
    float value;
    int ind, index_modifier, smaller_count=0;
    int pix_r = 2 * blockIdx.x + threadIdx.y;
    //int r_size = 2 * gridDim.x;
    int pix_c = blockIdx.y;
    int c_size = gridDim.y;

    // give the thread the value it will sort. Note that frames is 3D row-major
    // ind3 = threadIdx.x, dim3 = buffer_length (i.e. 32)
    //value = frames[threadIdx.x + 32 * (blockIdx.y + gridDim.y * (2 * blockIdx.x + threadIdx.y))];//frames[threadIdx.x + 32 * (pix_c + c_size * pix_r)];
    value = frames[threadIdx.x + 32 * (pix_c + c_size * pix_r)];

    // load everything into shared
    to_sort[shared_offset + threadIdx.x] = value;

    // zero the degeneracy so that adding one with make it 1, and if a second thread wants the same spot it will shift
    degeneracy_counter[shared_offset + threadIdx.x] = 0;

    // make sure shared allocations are loaded
    __syncthreads();

    // have each thread check sum the number of values less than the one it's responsible for
    for (ind = 0; ind < 32; ind++){
        if (to_sort[shared_offset + ind] < value){
            smaller_count += 1;
        }
    }

    //atomically add 1 to the index the thread thinks it goes to
    index_modifier = atomicAdd(&degeneracy_counter[shared_offset + smaller_count], 1);

    // make sure all threads are done with to_sort before we replace it
    __syncthreads();

    // write value into to_sort at the degeneracy-modified index.
    to_sort[shared_offset + smaller_count + index_modifier] = value;

    __syncthreads();

    // have one thread for each pixel write out nth value for that pixel
    if (threadIdx.x == 0){
        // nth_values is 2D row-major
//         nth_values[blockIdx.y + gridDim.y * (2* blockIdx.x + threadIdx.y)] = to_sort[shared_offset + n];
        nth_values[pix_c + c_size * pix_r] = to_sort[shared_offset + n];
    }

    // uncomment next part for debugging -> push sorted frames back for frames
    // DO NOT leave this uncommented for normal execution (will screw up which frames are actually on GPU)
//    frames[threadIdx.x + 32 * (blockIdx.y + gridDim.y * (2 * blockIdx.x + threadIdx.y))] = to_sort[shared_offset + threadIdx.x];
//    frames[threadIdx.x + 32 * (blockIdx.y + gridDim.y * (2 * blockIdx.x + threadIdx.y))] = index_modifier;

}


__global__ void nth_value_by_pixel_search_sort_dynamic(float *frames, const int n, const int spots_filled, const int buffer_length, float *nth_values)
/*
Same function as above, only shared memory is dynamically allocated
*/
{

    int shared_offset_to_sort = spots_filled * threadIdx.y; // offset since we're doing two pixels per block
    int shared_offset_degeneracy_counter = 2 * spots_filled + (spots_filled * threadIdx.y);
    extern __shared__ float sh_array[];
//    int * degeneracy_counter = (int*)sh_array;
//    float * degeneracy_counter = (float*)sh_array;
//    float * to_sort = (float*)(&degeneracy_counter[2 * buffer_length]);
//    extern __shared__ int degeneracy_counter[];
//    extern __shared__ float to_sort[];  // will be 2 * buffer length (e.g. 64)
//    __shared__ int degeneracy_counter[64];
//    __shared__ float to_sort[64];  // will be 2 * buffer length (e.g. 64)
    float value;
    int ind, index_modifier, smaller_count=0;
    int pix_r = 2 * blockIdx.x + threadIdx.y;
    //int r_size = 2 * gridDim.x;
    int pix_c = blockIdx.y;
    int c_size = gridDim.y;

    // give the thread the value it will sort. Note that frames is 3D row-major
    // ind3 = threadIdx.x, dim3 = buffer_length (i.e. 32)
    //value = frames[threadIdx.x + 32 * (blockIdx.y + gridDim.y * (2 * blockIdx.x + threadIdx.y))];//frames[threadIdx.x + 32 * (pix_c + c_size * pix_r)];
    value = frames[threadIdx.x + buffer_length * (pix_c + c_size * pix_r)];

//    if (threadIdx.x == 0 & threadIdx.y == 0){
//        printf("dc pointer: %f\n", to_sort[0]);
//    }

    // load everything into shared
//    to_sort[shared_offset + threadIdx.x] = value;
    sh_array[shared_offset_to_sort + threadIdx.x] = value;

    // zero the degeneracy so that adding one with make it 1, and if a second thread wants the same spot it will shift
//    degeneracy_counter[shared_offset + threadIdx.x] = 0;
    sh_array[shared_offset_degeneracy_counter + threadIdx.x] = 0;

    // make sure shared allocations are loaded
    __syncthreads();

    // have each thread check sum the number of values less than the one it's responsible for
    for (ind = 0; ind < spots_filled; ind++){
//        if (to_sort[shared_offset + ind] < value){
        if (sh_array[shared_offset_to_sort + ind] < value){
            smaller_count += 1;
        }
    }

    //atomically add 1 to the index the thread thinks it goes to
//    index_modifier = (int) atomicAdd(&degeneracy_counter[shared_offset + smaller_count], 1.0f);
    index_modifier = (int) atomicAdd(&sh_array[shared_offset_degeneracy_counter + smaller_count], 1);

    // make sure all threads are done with to_sort before we replace it
    __syncthreads();

    // write value into to_sort at the degeneracy-modified index.
//    to_sort[shared_offset + smaller_count + index_modifier] = value;
    sh_array[shared_offset_to_sort + smaller_count + index_modifier] = value;

    __syncthreads();
    // have one thread for each pixel write out nth value for that pixel
    if (threadIdx.x == 0){
        // nth_values is 2D row-major
//         nth_values[blockIdx.y + gridDim.y * (2* blockIdx.x + threadIdx.y)] = to_sort[shared_offset + n];
//        nth_values[pix_c + c_size * pix_r] = to_sort[shared_offset + n];
        nth_values[pix_c + c_size * pix_r] = sh_array[shared_offset_to_sort + n];
    }

    // uncomment next part for debugging -> push sorted frames back for frames
    // DO NOT leave this uncommented for normal execution (will screw up which frames are actually on GPU)
//   __syncthreads();
//   frames[threadIdx.x + buffer_length * (blockIdx.y + gridDim.y * (2 * blockIdx.x + threadIdx.y))] = to_sort[shared_offset + threadIdx.x];
//    frames[threadIdx.x + 32 * (blockIdx.y + gridDim.y * (2 * blockIdx.x + threadIdx.y))] = index_modifier;

}





// ---------------------------------------------------------------------------------------------- //
// ---------------------------------------------------------------------------------------------- //


// ------------------------------ DEPRECIATED CODE BELOW THIS LINE ------------------------------ //

// ---------------------------------------------------------------------------------------------- //
// ---------------------------------------------------------------------------------------------- //









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



}