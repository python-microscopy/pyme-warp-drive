import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <thrust/sort.h>


    extern "C" { // extern C required for PyCUDA to find our functions


    __global__ void nth_value_by_pixel(float *frames, const int n, float *nth_values)
    {
        int data_loc;
        float to_sort[30];

        for (int ind = 0; ind < 30; ind++){
            // row-major 3D index the same pixel in depth (fastest changing)
            data_loc = ind + 30 * (threadIdx.y + blockDim.y * threadIdx.x); 
            // threadIdx.x * col_size + threadIdx.y + ind * col_size * row_size;
            to_sort[ind] = frames[data_loc];
        }

        thrust::sort(thrust::device, to_sort, to_sort + 30);

        printf("test_val is %f", to_sort[n]);

        // should not need to sync threads
        nth_values[threadIdx.y + threadIdx.x * blockDim.y] = to_sort[n];
    }
    }
    """, no_extern_c=True)  # no_extern_c required to avoid compilation error if using modules like thrust
get_percentile = mod.get_function("nth_value_by_pixel")

# allocate and transfer input
pix_x = 2
pix_y = 3

frames = 10 * np.ones((pix_x, pix_y, 30))
frames[:, :, 10] = np.reshape(np.arange(0, pix_y * pix_x), (pix_x, pix_y))
print(frames[:, :, 10])
frames = frames.astype(np.float32)
frames_gpu = cuda.mem_alloc(frames.size * frames.dtype.itemsize)
cuda.memcpy_htod(frames_gpu, frames)

# allocate output

output = cuda.mem_alloc(pix_x * pix_y * frames.dtype.itemsize)
test = np.empty((pix_x, pix_y), dtype=np.float32)

# allocate test to_sort
to_sort = np.zeros(30, dtype=np.float32)
to_sort_gpu = cuda.mem_alloc(to_sort.size * to_sort.dtype.itemsize)

# get_percentile(frames_gpu, np.int32(pix_x), np.int32(pix_y), np.int32(29), output, to_sort_gpu, block=(pix_x, pix_y, 1))
n_to_grab = 0

# get_percentile(frames_gpu, np.int32(n_to_grab), output, to_sort_gpu, block=(pix_x, pix_y, 1))
get_percentile(frames_gpu, np.int32(n_to_grab), output, block=(pix_x, pix_y, 1))
cuda.memcpy_dtoh(test, output)
print(test)