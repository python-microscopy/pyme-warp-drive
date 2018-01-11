import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import numpy as np
from pycuda.compiler import SourceModule




class Buffer(object):
    def __init__(self, data_buffer, percentile=0.25, buffer_length=30, threads_per_block_in_sort=64):
        self.data_buffer = data_buffer
        self.buffer_length = buffer_length
        self.percentile = percentile

        self.cur_frames = set()
        self.cur_positions = {}
        self.available = range(buffer_length)

        self.slice_shape = self.data_buffer.dataSource.getSliceShape()

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

        # get info about selected GPU
        self.gpu_info = pycuda.tools.DeviceData()

        self.threads_per_block_in_sort = threads_per_block_in_sort
        # determine how many warps can fit for our given buffer length
        shared_mem_per_block = self.frames.itemsize * threads_per_block_in_sort * buffer_length
        # print 'Shared Memory size:', info.shared_memory
        # print 'Blocks per MP:', info.thread_blocks_per_mp
        # print 'MP count:', self.dev.multiprocessor_count

        #---- compile
        print('compiling!\n')
        self.compile()

    def compile(self):
        mod = SourceModule("""
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
            
            }
            """, no_extern_c=True)#, options=["--maxrregcount=32"])  # no_extern_c required to avoid compilation error if using modules like thrust
        self.nth_value_by_pixel = mod.get_function('nth_value_by_pixel')
        self.nth_value_by_pixel_shared_quicksort = mod.get_function('nth_value_by_pixel_shared_quicksort')
        self.clear_frames = mod.get_function('clear_frames')
        self.update_frame = mod.get_function('update_frame')

    def update(self, frame, position):
        # send new frame to GPU
        cuda.memcpy_htod(self.new_frame_gpu, self.data_buffer.dataSource.getSlice(frame).astype(np.float32))
        # update the frame buffer on the GPU
        self.update_frame(self.frames_gpu, self.new_frame_gpu, position,
                          block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1))

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
        bg_indices = set(bg_indices)
        if bg_indices == self.cur_frames:
            return self.cur_bg
        else:
            self.update_buffer(bg_indices)

            # determine which index we want
            n = round(self.percentile * len(bg_indices))

            # self.nth_value_by_pixel(self.frames_gpu, np.int32(n), self.cur_bg_gpu,
            #                         block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1))
            # # TODO - write nth_value_by_pixel_32 to divy up pixels into warps and run 1 warp per block
            # # can only fit 15 or less warps per SM at a time, else we exceed maximum shared memory per multiprocessor for -50:0 bg_indices
            warp_count_x = int(np.ceil(self.slice_shape[0] / 32.0))
            self.nth_value_by_pixel_shared_quicksort(self.frames_gpu, np.int32(n), self.cur_bg_gpu,
                                                     block=(32, 1, 1), grid=(warp_count_x, self.slice_shape[1]), shared=32*self.buffer_length*self.frames.itemsize)

            # copy value back to host
            cuda.memcpy_dtoh(self.cur_bg, self.cur_bg_gpu)

            return self.cur_bg


class SimpleBuffer_CPU(object):
    def __init__(self, data_buffer, percentile=0.25, buffer_length=30):
        self.data_buffer = data_buffer
        self.percentile = percentile
        self.buffer_length=buffer_length

        self.slice_shape = self.data_buffer.dataSource.getSliceShape()

        self.frames = np.empty((self.slice_shape[0], self.slice_shape[1], self.buffer_length))

        self.cur_frames = set()
        self.cur_bg = None

        self.index_of_interest = round(percentile * g_buf.buffer_length)

    def getBackground(self, bg_indices):
        if bg_indices == self.cur_frames:
            return self.cur_bg
        else:
            for fi in range(g_buf.buffer_length):
                cpu_buffer[:, :, fi] = ds.getSlice(fi)

            cpu_sorted = np.sort(cpu_buffer, axis=2)

            self.cur_bg = cpu_sorted[:,:,index_of_interest]

            return self.cur_bg


class dbuff(object):
    pass

if __name__ == '__main__':
    from PYME.IO.DataSources.RandomDataSource import DataSource
    percentile = 0.25
    # run a test
    imsz= 1024
    ds = DataSource(imsz, imsz, 100)
    dbuff.dataSource = ds
    g_buf = Buffer(dbuff, percentile=percentile)

    bg_gpu = g_buf.getBackground(set(range(30)))

    # check if this is also what the CPU gets
    cpu_buffer = np.empty((imsz, imsz, g_buf.buffer_length))
    for fi in range(g_buf.buffer_length):
        cpu_buffer[:,:,fi] = dbuff.dataSource.getSlice(fi)
    cpu_sorted = np.sort(cpu_buffer, axis=2)
    index_of_interest = round(percentile*g_buf.buffer_length)
    bg_cpu = cpu_sorted[:,:,index_of_interest]

    success = np.array_equal(bg_cpu, bg_gpu)
    print('test passed: %r' % success)





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
    benchmark = False
    if benchmark:
        import timeit
        setup_script = """
        class buff(object):
            pass
        
        from PYME.IO.DataSources.RandomDataSource import DataSource
        from warpDrive.buffers import Buffer
        
        percentile = 0.25
        # run a test
        imsz = 3
        ds = DataSource(imsz, imsz, 100)
        buff.dataSource = ds
        g_buf = Buffer(buff, percentile=percentile)
        indices = set(range(30))
        """
        timeit.timeit('g_buf.getBackground(indices)', setup=setup_script, number=1000)




        setup_cpu = """
        class buff(object):
            pass
        
        from PYME.IO.DataSources.RandomDataSource import DataSource
        
        percentile = 0.25
        # run a test
        imsz = 3
        ds = DataSource(imsz, imsz, 100)
        
        dbuff = buff()
        dbuff.dataSource = ds
        
        dbuff.getSlice = ds.getSlice
        from PYME.IO.buffers import backgroundBufferM
        c_buf = backgroundBufferM(dbuff, percentile=percentile)
        indices = set(range(30))
        """

        timeit.timeit('c_buf.getBackground(indices)', setup=setup_cpu, number=1000)
