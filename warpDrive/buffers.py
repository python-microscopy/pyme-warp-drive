import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import numpy as np
from pycuda.compiler import SourceModule




class Buffer(object):
    def __init__(self, data_buffer, percentile=0.25, buffer_length=30):
        self.data_buffer = data_buffer
        self.buffer_length = buffer_length
        self.percentile = percentile

        self.cur_frames = set()
        self.cur_positions = {}
        self.available = range(buffer_length)

        self.slice_shape = self.data_buffer.getSliceShape()

        #---- allocate memory
        pix_r, pix_c = self.slice_shape
        self.cur_bg = np.empty((pix_r, pix_c), np.float32)
        self.cur_bg_gpu = cuda.mem_alloc(pix_r * pix_c * self.cur_bg.dtype.itemsize)

        self.frames = np.empty((pix_r, pix_c, self.buffer_length), np.float32)
        self.frames_gpu = cuda.mem_alloc(self.frames.size * self.frames.dtype.itemsize)
        # a = self.frames_gpu.as_buffer()

        self.to_wipe = np.empty((self.buffer_length), np.float32)
        self.to_wipe_gpu = cuda.mem_alloc(self.to_wipe.size * self.to_wipe.dtype.itemsize)

        self.new_frame_gpu = cuda.mem_alloc(pix_r * pix_c * self.cur_bg.dtype.itemsize)

        #---- compile
        self.compile()

    def compile(self):
        mod = SourceModule("""
            #include <thrust/sort.h>


            extern "C" { // extern C required for PyCUDA to find our functions


            __global__ void nth_value_by_pixel(float *frames, const int n, float *nth_values)
            /*
                To be called with block dim: (image.shape[0], image.shape[1], 1)
            */
            {
                int data_loc;
                float to_sort[30];

                for (int ind = 0; ind < 30; ind++){
                    // row-major 3D index the same pixel in depth (fastest changing)
                    data_loc = ind + 30 * (threadIdx.y + blockDim.y * threadIdx.x); 

                    to_sort[ind] = frames[data_loc];
                }

                thrust::sort(thrust::device, to_sort, to_sort + 30);

                printf("test_val is %f", to_sort[n]);

                // should not need to sync threads
                nth_values[threadIdx.y + threadIdx.x * blockDim.y] = to_sort[n];
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
                To be called with block dim: (image.shape[0], image.shape[1], 1)
            */
            {
                int data_loc;

                // row-major 3D index the same pixel in depth (fastest changing)
                data_loc = frame_num + 30 * (threadIdx.y + blockDim.y * threadIdx.x); 
                
                // replace value with new frame value
                frames[data_loc] = new_frame[threadIdx.y + threadIdx.x * blockDim.y];
            }
            
            }
            """, no_extern_c=True)  # no_extern_c required to avoid compilation error if using modules like thrust
        self.nth_value_by_pixel = mod.get_function('nth_value_by_pixel')
        self.clear_frames = mod.get_function('clear_frames')
        self.update_frame = mod.get_function('update_frame')

    def update(self, frame, position):
        # send new frame to GPU
        cuda.memcpy_htod(self.new_frame_gpu, self.data_buffer.getSlice(frame).astype(np.float32))

        # update the frame buffer on the GPU
        self.update_frame(self.frames_gpu, self.new_frame_gpu, position,
                          block=(self.slice_shape[0], self.slice_shape[1], 1))

        # update position dict
        self.cur_positions[frame] = position

    def update_buffer(self, bg_indices):
        if len(bg_indices) > self.buffer_length:
            raise RuntimeError('GPU frame buffer is too small for background indices selected')

        fresh = bg_indices - self.cur_frames
        uncleared = self.cur_frames - bg_indices

        # make sure we have all the current frames on the CPU
        for frame in fresh:
            # if we have unwanted frames on the GPU, replace them
            if len(uncleared) > 0:
                # find where to put new frame, and clean up the position dictionary in the process
                position = self.cur_positions.pop(uncleared.pop())

            else:
                # need to add data without replacing anything
                position = np.int32(self.available.pop())

            self.update(frame, position)

        # clear unwanted frames still living on the GPU
        num_excess = len(uncleared)
        if num_excess > 0:
            # update frames to be wiped on GPU (note that only :num_excesss matter)
            self.to_wipe[:num_excess] = [self.cur_positions.pop(uncleared.pop()) for dummy in range(num_excess)]
            cuda.memcpy_htod(self.to_wipe_gpu, self.to_wipe)

            # clear
            self.clear_frames(self.frames_gpu, self.to_wipe_gpu,
                              block=(self.slice_shape[0], self.slice_shape[1], num_excess))

            # update available list
            self.available.append(self.to_wipe[:num_excess])

        # update current frame set
        self.cur_frames = bg_indices  # at this point, we've added all the new frames and removed all the outdated ones

    def getBackground(self, bg_indices):
        if bg_indices == self.cur_frames:
            return self.cur_bg
        else:
            self.update_buffer(bg_indices)

            # determine which index we want
            n = round(self.percentile * len(bg_indices))

            self.nth_value_by_pixel(self.frames_gpu, np.int32(n), self.cur_bg_gpu,
                                    block=(self.slice_shape[0], self.slice_shape[1], 1))

            # copy value back to host
            cuda.memcpy_dtoh(self.cur_bg, self.cur_bg_gpu)

            return self.cur_bg


if __name__ == '__main__':
    from PYME.IO.DataSources.RandomDataSource import DataSource

    # run a test
    ds = DataSource(3, 3, 100)
    g_buf = Buffer(ds)

    g_buf.getBackground(set(range(30)))



    # # allocate test to_sort
    # to_sort = np.zeros(30, dtype=np.float32)
    # to_sort_gpu = cuda.mem_alloc(to_sort.size * to_sort.dtype.itemsize)
    #
    # # get_percentile(frames_gpu, np.int32(pix_x), np.int32(pix_y), np.int32(29), output, to_sort_gpu, block=(pix_x, pix_y, 1))
    # n_to_grab = 0
    #
    # # get_percentile(frames_gpu, np.int32(n_to_grab), output, to_sort_gpu, block=(pix_x, pix_y, 1))
    # get_percentile(frames_gpu, np.int32(n_to_grab), output, block=(pix_x, pix_y, 1))
    # cuda.memcpy_dtoh(test, output)
    # print(test)
