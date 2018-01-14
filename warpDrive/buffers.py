import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import numpy as np
from pycuda.compiler import SourceModule
import buffers_cu



class Buffer(object):
    def __init__(self, data_buffer, percentile=0.25, buffer_length=30, dark_map=None):
        self.data_buffer = data_buffer
        self.buffer_length = buffer_length
        self.percentile = percentile

        self.index_to_grab = np.int32(round(self.percentile * buffer_length))

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
        # self.gpu_info = pycuda.tools.DeviceData()
        #
        # self.threads_per_block_in_sort = threads_per_block_in_sort
        # # determine how many warps can fit for our given buffer length
        # shared_mem_per_block = self.frames.itemsize * threads_per_block_in_sort * buffer_length
        # print 'Shared Memory size:', info.shared_memory
        # print 'Blocks per MP:', info.thread_blocks_per_mp
        # print 'MP count:', self.dev.multiprocessor_count

        # create stream so background can be estimated asynchronously
        self.bg_streamer = cuda.Stream()

        # dark map hack
        self.dark_map_gpu = None
        if dark_map is not None:
            if np.isscalar(dark_map):
                dark_map = dark_map * np.ones((self.slice_shape))

            self.dark_map_gpu = cuda.mem_alloc(pix_r * pix_c * self.cur_bg.dtype.itemsize)
            cuda.memcpy_htod(self.dark_map_gpu, dark_map.astype(np.float32))

        #---- compile
        print('compiling!\n')
        self.compile()

    def compile(self):
        mod = buffers_cu.percentile_buffer()
        self.nth_value_by_pixel = mod.get_function('nth_value_by_pixel')
        self.nth_value_by_pixel_shared_quicksort = mod.get_function('nth_value_by_pixel_shared_quicksort')
        self.clear_frames = mod.get_function('clear_frames')
        self.update_frame = mod.get_function('update_frame')
        self.subtract_b_from_a = mod.get_function('subtract_b_from_a')

    def update(self, frame, position):
        # send new frame to GPU
        cuda.memcpy_htod_async(self.new_frame_gpu,
                               np.ascontiguousarray(self.data_buffer.dataSource.getSlice(frame), dtype=np.float32),
                               stream=self.bg_streamer)
        # update the frame buffer on the GPU
        self.update_frame(self.frames_gpu, self.new_frame_gpu, position,
                          block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1), stream=self.bg_streamer)

        # update position dict
        self.cur_positions[frame] = position

        # NB - this function returns without synchronizing the stream

    def update_buffer(self, bg_indices):
        if len(bg_indices) != self.buffer_length:
            raise RuntimeError('changing GPU background buffer size is not currently supported')

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
            cuda.memcpy_htod_async(self.to_wipe_gpu, self.to_wipe, stream=self.bg_streamer)

            # clear
            self.clear_frames(self.frames_gpu, self.to_wipe_gpu,
                              block=(self.slice_shape[0], self.slice_shape[1], num_excess), stream=self.bg_streamer)

            # update available list
            self.available.append(self.to_wipe[:num_excess])

        # update current frame set
        self.cur_frames = bg_indices  # at this point, we've added all the new frames and removed all the outdated ones

    def calc_background(self, bg_indices, subtract_dark_map=True):
        """
        Just calculates the background, does not return it, nor does it wait for the calculation to terminate
        :param bg_indices:
        :return:
        """
        bg_indices = set(bg_indices)

        self.update_buffer(bg_indices)

        # self.nth_value_by_pixel(self.frames_gpu, np.int32(n), self.cur_bg_gpu,
        #                         block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1))

        warp_count_x = int(np.ceil(self.slice_shape[0] / 32.0))
        self.nth_value_by_pixel_shared_quicksort(self.frames_gpu, self.index_to_grab, self.cur_bg_gpu,
                                                 block=(32, 1, 1), grid=(warp_count_x, self.slice_shape[1]),
                                                 shared=32 * self.buffer_length * self.frames.itemsize,
                                                 stream=self.bg_streamer)

        if subtract_dark_map and self.dark_map_gpu:
            self.subtract_b_from_a(self.cur_bg_gpu, self.dark_map_gpu,
                                   block=(32, 1, 1), grid=(warp_count_x, self.slice_shape[1]), stream=self.bg_streamer)

        # NB - this function does not wait for the calculation to finish before returning, and does not copy anything
        # back to the GPU!


    def sync_calculation(self):
        """
        synchronize CUDA stream we are calculating the background in
        :return:
        """
        self.bg_streamer.synchronize()

    def getBackground(self, bg_indices):
        # estimate the background
        self.calc_background(bg_indices, subtract_dark_map=False)
        # bring it back from the GPU
        cuda.memcpy_dtoh_async(self.cur_bg, self.cur_bg_gpu, stream=self.bg_streamer)

        # make sure that calculation has finished and background has been passed back from the GPU
        self.sync_calculation()

        return self.cur_bg

    def get_background(self):
        self.sync_calculation()
        return self.cur_bg

    def getBackground_depreciated(self, bg_indices):
        """
        Does the full update and calculation
        :param bg_indices:
        :return:
        """
        bg_indices = set(bg_indices)
        if bg_indices == self.cur_frames:
            return self.cur_bg
        else:
            self.update_buffer(bg_indices)

            # determine which index we want
            n = round(self.percentile * len(bg_indices))

            # self.nth_value_by_pixel(self.frames_gpu, np.int32(n), self.cur_bg_gpu,
            #                         block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1))
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

def main():
    from PYME.IO.DataSources.RandomDataSource import DataSource
    percentile = 0.25
    # run a test
    imsz = 1024
    ds = DataSource(imsz, imsz, 100)
    dbuff.dataSource = ds
    g_buf = Buffer(dbuff, percentile=percentile)

    bg_gpu = g_buf.getBackground(set(range(30)))

    # check if this is also what the CPU gets
    cpu_buffer = np.empty((imsz, imsz, g_buf.buffer_length))
    for fi in range(g_buf.buffer_length):
        cpu_buffer[:, :, fi] = dbuff.dataSource.getSlice(fi)
    cpu_sorted = np.sort(cpu_buffer, axis=2)
    index_of_interest = round(percentile * g_buf.buffer_length)
    bg_cpu = cpu_sorted[:, :, index_of_interest]

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
        setup_script = """class buff(object):
            pass

        from PYME.IO.DataSources.RandomDataSource import DataSource
        from warpDrive.buffers import Buffer

        percentile = 0.25
        # run a test
        ds = DataSource(960, 240, 100)
        buff.dataSource = ds
        g_buf = Buffer(buff, percentile=percentile)
        indices = set(range(30))
        """
        timeit.timeit('g_buf.getBackground(indices)', setup=setup_script, number=1000)

        setup_script = """class buff(object):
            pass

        from PYME.IO.DataSources.RandomDataSource import DataSource
        from warpDrive.buffers import Buffer

        percentile = 0.25
        # run a test
        ds = DataSource(960, 240, 100)
        buff.dataSource = ds
        g_buf = Buffer(buff, percentile=percentile)
        g_buf.getBackground(set(range(30)))
        indices = set(range(1,31))
        """
        timeit.timeit('g_buf.getBackground(indices)', setup=setup_script, number=1000)

        setup_cpu = """
        class buff(object):
            pass

        from PYME.IO.DataSources.RandomDataSource import DataSource

        percentile = 0.25
        # run a test
        ds = DataSource(960, 240, 100)

        dbuff = buff()
        dbuff.dataSource = ds

        dbuff.getSlice = ds.getSlice
        from PYME.IO.buffers import backgroundBufferM
        c_buf = backgroundBufferM(dbuff, percentile=percentile)
        indices = set(range(30))
        """

        timeit.timeit('c_buf.getBackground(indices)', setup=setup_cpu, number=1000)

        setup_cpu = """
            class buff(object):
                pass

            from PYME.IO.DataSources.RandomDataSource import DataSource

            percentile = 0.25
            # run a test
            ds = DataSource(960, 240, 100)

            dbuff = buff()
            dbuff.dataSource = ds

            dbuff.getSlice = ds.getSlice
            from PYME.IO.buffers import backgroundBufferM
            c_buf = backgroundBufferM(dbuff, percentile=percentile)
            c_buf.getBackground(set(range(30)))
            indices = set(range(1, 31))
            """

        timeit.timeit('c_buf.getBackground(indices)', setup=setup_cpu, number=1000)

if __name__ == '__main__':
    main()

