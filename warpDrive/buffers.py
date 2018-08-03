# Andrew Barentine, andrew.barentine@yale.edu

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import buffers_cu

# It is convenient in the python-microsocpy environment (PYME) if this buffer is a subclass, but to keep things somewhat
# independent, we can fall back to subclassing object if PYME is not available
try:
    from PYME.IO.buffers import backgroundBufferM as to_subclass
except ImportError:
    to_subclass = object
    raise RuntimeWarning('Cannot import python-microscopy environment (PYME) background buffer - this buffer might ' 
                         'not interface correctly with PYME')


class Buffer(to_subclass):
    """
    Handler for GPU-based percentile buffering
    """
    def __init__(self, data_buffer, percentile=0.25, buffer_length=32, dark_map=None):
        self.data_buffer = data_buffer
        self.buffer_length = buffer_length
        self.percentile = percentile

        self.index_to_grab = np.int32(max([round(self.percentile * buffer_length) - 1, 0]))

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

        # self.to_wipe = np.empty((self.buffer_length), np.float32)
        # self.to_wipe_gpu = cuda.mem_alloc(self.to_wipe.size * self.to_wipe.dtype.itemsize)

        self.new_frame_gpu = cuda.mem_alloc(pix_r * pix_c * self.cur_bg.dtype.itemsize)

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
        """

        Compiles CUDA functions using PyCUDA SourceModule

        Returns
        -------
        Nothing

        """
        mod = buffers_cu.percentile_buffer()
        self.nth_value_by_pixel = mod.get_function('nth_value_by_pixel')
        self.nth_value_by_pixel_shared_quicksort = mod.get_function('nth_value_by_pixel_shared_quicksort')
        self.nth_value_by_pixel_search_sort = mod.get_function('nth_value_by_pixel_search_sort')
        self.nth_value_by_pixel_search_sort_dynamic = mod.get_function('nth_value_by_pixel_search_sort_dynamic')
        self.clear_frame = mod.get_function('clear_frame')
        self.update_frame = mod.get_function('update_frame')
        self.subtract_b_from_a = mod.get_function('subtract_b_from_a')

    def update(self, frame, position, frame_data):
        """

        Asynchronously replaces the frame currently residing on the GPU in the 'position' slice with the new frame. This
        is performed in the bg_streamer stream, which is NOT synchronized before this functions returns.

        Parameters
        ----------
        frame : int
            time-index of frame being added to the buffer
        position : int
            position index where frame will be inserted in the buffer array held on the GPU.
        frame_data : ndarray
            frame data as numpy.float32, contiguous in memory
        Returns
        -------
        Nothing

        """
        # send new frame to GPU
        cuda.memcpy_htod_async(self.new_frame_gpu, frame_data, stream=self.bg_streamer)
        # update the frame buffer on the GPU
        self.update_frame(self.frames_gpu, self.new_frame_gpu, position, np.int32(self.buffer_length),
                          block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1), stream=self.bg_streamer)

        # update position dict
        self.cur_positions[frame] = position

        # NB - this function returns without synchronizing the stream

        # NEXT LINE FOR DEBUGGING ONLY
        # cuda.memcpy_dtoh(self.frames, self.frames_gpu)

    def clear(self, frame):
        """


        """
        position = self.cur_positions.pop(frame)

        # update the frame buffer on the GPU
        self.clear_frame(self.frames_gpu, position, np.int32(self.buffer_length),
                          block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1), stream=self.bg_streamer)

        # mark this position as available
        self.available.append(position)
        # NB - this function returns without synchronizing the stream


    def update_buffer(self, bg_indices):
        """

        Loops through bg_indices and ensures the desired frames are on the GPU. Note that the position where each slice
        is stored on the GPU does not matter.

        Parameters
        ----------
        bg_indices : set
            Frame indices to use when estimating the background for the frame about to be fit

        Returns
        -------
        Nothing

        """
        if len(bg_indices) > self.buffer_length:
            raise RuntimeError('changing GPU background buffer size is not currently supported')

        cur_frames = set(self.cur_positions.keys())

        fresh = bg_indices - cur_frames
        uncleared = cur_frames - bg_indices

        # # reset buffer if it is outdated
        # if len(uncleared) == self.buffer_length:
        #     uncleared = set()
        #     self.available = range(self.buffer_length)

        # make sure we have all the current frames on the CPU
        for frame in fresh:
            frame_data = np.ascontiguousarray(self.data_buffer.getSlice(frame), dtype=np.float32)
            # if we have unwanted frames on the GPU, replace them
            if len(uncleared) > 0:
                # find where to put new frame, and clean up the position dictionary in the process
                position = self.cur_positions.pop(uncleared.pop())

            else:  # need to add data without replacing anything
                # start from the front to make indexing on the GPU easier for un-full buffers
                position = np.int32(self.available.pop(0))

            self.update(frame, position, frame_data)

        # clear unwanted frames still living on the GPU
        for frame in uncleared:
            self.clear(frame)
        # num_excess = len(uncleared)
        # if num_excess > 0:
        #     # update frames to be wiped on GPU (note that only :num_excesss matter)
        #     self.to_wipe[:num_excess] = [self.cur_positions.pop(uncleared.pop()) for dummy in range(num_excess)]
        #     cuda.memcpy_htod_async(self.to_wipe_gpu, self.to_wipe, stream=self.bg_streamer)
        #
        #     # clear
        #     self.clear_frames(self.frames_gpu, self.to_wipe_gpu,
        #                       block=(self.slice_shape[0], self.slice_shape[1], num_excess), stream=self.bg_streamer)
        #
        #     # update available list
        #     self.available.append(self.to_wipe[:num_excess])


        # update current frame set
        # self.cur_frames = bg_indices  # at this point, we've added all the new frames and removed all the outdated ones

    def calc_background(self, bg_indices, subtract_dark_map=True):
        """

        Just calculates the background, does not return it, nor does it wait for the calculation to terminate before
        returning. This allows us to calculate the background asynchronously and later call Buffer.sync_calculation() to
        make sure everything is finished before using Buffer.cur_bg_gpu or pulling it back to the CPU.

        Parameters
        ----------
        bg_indices : iterable
            Frame indices to use when estimating the background for the frame about to be fit
        subtract_dark_map : bool
            Flag to subtract the dark map from the background estimate while it is still on the GPU.

        Returns
        -------

        """
        bg_indices = set(bg_indices)

        self.update_buffer(bg_indices)

        # block=(buffer_length, 2), grid=(slice_size[0] / 2, slice_size[1])
        num_indices = len(bg_indices)
        full = num_indices == self.buffer_length
        if full and self.buffer_length == 32:
            # no need to dynamically allocate shared memory
            self.nth_value_by_pixel_search_sort(self.frames_gpu, self.index_to_grab, self.cur_bg_gpu,
                                                block=(self.buffer_length, 2, 1),
                                                grid=(self.slice_shape[0] / 2, self.slice_shape[1]),
                                                stream=self.bg_streamer)
        elif full:
            # need to dynamically allocate
            filled = np.int32(self.buffer_length)
            self.nth_value_by_pixel_search_sort_dynamic(self.frames_gpu, self.index_to_grab,
                                                        filled, filled, self.cur_bg_gpu,
                                                        block=(self.buffer_length, 2, 1),
                                                        grid=(self.slice_shape[0] / 2, self.slice_shape[1]),
                                                        stream=self.bg_streamer,
                                                        shared=2 * 2 * self.buffer_length * 4)
        else:
            # buffer is not full, need to change which index we grab and dynamically allocate shared
            index_to_grab = np.int32(max([round(self.percentile * num_indices) - 1, 0]))
            filled = np.int32(num_indices)
            self.nth_value_by_pixel_search_sort_dynamic(self.frames_gpu, index_to_grab,
                                                        filled, np.int32(self.buffer_length), self.cur_bg_gpu,
                                                        block=(num_indices, 2, 1),
                                                        grid=(self.slice_shape[0] / 2, self.slice_shape[1]),
                                                        stream=self.bg_streamer,
                                                        shared=2 * 2 * num_indices * 4)




        if subtract_dark_map and self.dark_map_gpu:
            warp_count_x = int(np.ceil(self.slice_shape[0] / 32.0))
            self.subtract_b_from_a(self.cur_bg_gpu, self.dark_map_gpu,
                                   block=(32, 1, 1), grid=(warp_count_x, self.slice_shape[1]), stream=self.bg_streamer)

        # NB - this function does not wait for the calculation to finish before returning, and does not copy anything
        # back to the GPU!

    def calc_background_quicksort(self, bg_indices, subtract_dark_map=True):
        """

        Depreciated version of Buffer.calc_background using a quicksort algorithm. This is not ideal due to the high
        thread divergence!

        """
        raise DeprecationWarning('This function is roughly an order of magnitude slower than Buffer.calc_background')
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

        Synchronizes the CUDA stream we are calculating the background in

        Returns
        -------
        Nothing

        """
        self.bg_streamer.synchronize()

    def getBackground(self, bg_indices):
        """

        Calculates the background and waits for the calculation to terminate before returning the background estimate.
        This function is useful if you need the background estimation performed in a non-asynchronous fashion.

        Parameters
        ----------
        bg_indices : iterable
            Frame indices to use when estimating the background for the frame about to be fit

        Returns
        -------
        cur_bg : ndarray
            Background estimate, based on the self.pctile percent of the frames indicated by bg_indices

        """

        # estimate the background
        self.calc_background(bg_indices, subtract_dark_map=False)
        # bring it back from the GPU
        cuda.memcpy_dtoh_async(self.cur_bg, self.cur_bg_gpu, stream=self.bg_streamer)

        # make sure that calculation has finished and background has been passed back from the GPU
        self.sync_calculation()

        return self.cur_bg

    def get_background(self):
        """

        Can be called AFTER Buffer.calc_background to transfer to background back to the CPU after the calculation
        completes.

        Returns
        -------
        cur_bg : ndarray
            Background estimate, based on the self.pctile percent of the frames indicated by bg_indices

        """
        cuda.memcpy_dtoh_async(self.cur_bg, self.cur_bg_gpu, stream=self.bg_streamer)
        self.sync_calculation()
        return self.cur_bg



# ----------- main() for testing GPU buffer sorting -----------



class dbuff(object):
    # TODO - move this dummy object into a proper unit test script along with main()
    pass

def main():
    from PYME.IO.DataSources.RandomDataSource import DataSource
    percentile = 0.25
    # run a test
    imsz_r = 960
    imsz_c = 240
    buffer_length = 31
    indices = set(range(7))

    ds = DataSource(imsz_r, imsz_c, 100)
    dbuff.dataSource = ds
    g_buf = Buffer(dbuff, percentile=percentile, buffer_length=buffer_length)

    bg_gpu = g_buf.getBackground(indices)

    # check if this is also what the CPU gets
    cpu_buffer = np.empty((imsz_r, imsz_c, g_buf.buffer_length))
    for fi in sorted(indices):#g_buf.buffer_length):
        cpu_buffer[:, :, fi] = dbuff.dataSource.getSlice(fi)
    cpu_sorted = np.sort(cpu_buffer[:,:,:len(indices)], axis=2)
    index_to_grab = np.int32(max([round(percentile * len(indices)) - 1, 0]))
    bg_cpu = cpu_sorted[:, :, index_to_grab]

    success = np.array_equal(bg_cpu, bg_gpu)
    print('test passed: %r' % success)


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
        g_buf = Buffer(buff, percentile=percentile, buffer_length=32)
        indices = set(range(32))
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
        g_buf.getBackground(set(range(32)))
        indices = set(range(1,33))
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
        indices = set(range(32))
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
        c_buf.getBackground(set(range(32)))
        indices = set(range(1, 33))
        """

        timeit.timeit('c_buf.getBackground(indices)', setup=setup_cpu, number=1000)

if __name__ == '__main__':
    main()

