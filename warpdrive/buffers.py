# Andrew Barentine, andrew.barentine@yale.edu

import pycuda.driver as cuda
#from pycuda import tools
import pycuda.autoinit
from pycuda import tools
import atexit
import numpy as np
from . import buffers_cu
import logging
logger = logging.getLogger(__name__)

# It is convenient in the python-microsocpy environment (PYME) if this buffer is a subclass, but to keep things somewhat
# independent, we can fall back to subclassing object if PYME is not available
try:
    from PYME.IO.buffers import backgroundBufferM as to_subclass
except ImportError:
    to_subclass = object
    raise RuntimeWarning('Cannot import python-microscopy environment (PYME) background buffer - this buffer might ' 
                         'not interface correctly with PYME')

global COMPILED_MODULE
logging.debug('compiling percentile buffer module')
COMPILED_MODULE = buffers_cu.percentile_buffer()


class Buffer(to_subclass):
    """
    Handler for GPU-based percentile buffering
    """
    def __init__(self, data_buffer, percentile=0.25, buffer_length=32, darkmap=None, flatmap=None,
                 electrons_per_count=None):
        self.data_buffer = data_buffer
        self.buffer_length = buffer_length
        self.percentile = percentile

        self.index_to_grab = np.int32(max([round(self.percentile * buffer_length) - 1, 0]))

        self.cur_frames = set()
        self.cur_positions = {}
        self.available = list(range(buffer_length))

        # create stream so background can be estimated asynchronously
        self.bg_streamer = cuda.Stream()

        # cuda.mem_alloc expects python int; avoid potential np.int64
        self.slice_shape = [int(d) for d in self.data_buffer.dataSource.getSliceShape()]

        #---- allocate memory
        pix_r, pix_c = self.slice_shape
        self.cur_bg = np.empty((pix_r, pix_c), np.float32)
        self.cur_bg_gpu = cuda.mem_alloc(pix_r * pix_c * self.cur_bg.dtype.itemsize)

        # TODO - ideally can initialize as empty, but inf is a part of hacky fix to pass test_recycling_after_IOError
        self.frames = np.inf * np.ones((pix_r, pix_c, self.buffer_length), 
                                       np.float32)
        self.frames_gpu = cuda.mem_alloc(self.frames.size * self.frames.dtype.itemsize)
        cuda.memcpy_htod_async(self.frames_gpu, self.frames, stream=self.bg_streamer)  # TODO - remove, part of hacky fix to pass test_recycling_after_IOError

        # a = self.frames_gpu.as_buffer()

        # self.to_wipe = np.empty((self.buffer_length), np.float32)
        # self.to_wipe_gpu = cuda.mem_alloc(self.to_wipe.size * self.to_wipe.dtype.itemsize)

        self.new_frame_gpu = cuda.mem_alloc(pix_r * pix_c * self.cur_bg.dtype.itemsize)

        # camera correction ~hack. TODO make gpu camera map manager
        self.electrons_per_count = np.float32(electrons_per_count)
        self.darkmap_gpu, self.flatmap_gpu = None, None
        map_mem_size = pix_r * pix_c * self.cur_bg.dtype.itemsize
        if darkmap is not None:
            if np.isscalar(darkmap):
                darkmap = darkmap * np.ones((self.slice_shape))

            self.darkmap_gpu = cuda.mem_alloc(map_mem_size)
            cuda.memcpy_htod(self.darkmap_gpu, np.ascontiguousarray(darkmap, dtype=np.float32))

        if flatmap is not None:
            if np.isscalar(flatmap):
                flatmap = flatmap * np.ones((self.slice_shape))

            self.flatmap_gpu = cuda.mem_alloc(map_mem_size)
            cuda.memcpy_htod(self.flatmap_gpu, np.ascontiguousarray(flatmap, dtype=np.float32))


        #---- get compiled function handles
        self._get_compiled_modules()
    
    def refresh_settings(self, percentile, buffer_length):
        """Make sure the Buffer instance has the desired percentile and buffer
        length

        Parameters
        ----------
        percentile : float
            fractional index to grab at each pixel (i.e. 0.5 corresponds to 
            the median)
        buffer_length : int
            max number of frames that fit in the buffer and the per-xy-pixel
            percentile is calculated on.
        
        Notes
        -----
        Buffer instance must maintain the same xy shape, for the same
        datasource, with the same dark and flatfield maps. Otherwise, 
        instantiate a new Buffer!
        """
        if percentile != self.percentile:
            logger.debug('changing percentile: %.3f -> %.3f' % (self.percentile,
                                                                percentile))
            self.percentile = percentile
            self.index_to_grab = np.int32(max([round(self.percentile * buffer_length) - 1, 0]))     

        if buffer_length != self.buffer_length:
            logger.debug('changing buffer length: %d -> %d' % (self.buffer_length,
                                                               buffer_length))
            self.buffer_length = buffer_length
            # changing the buffer length changes the index to grab
            self.index_to_grab = np.int32(max([round(self.percentile * buffer_length) - 1, 0])) 

            self.cur_frames = set()
            self.cur_positions = {}
            self.available = list(range(buffer_length))
            
            # reallocate frames arrays
            pix_r, pix_c = self.slice_shape
            # TODO - ideally can initialize as empty, but inf is a part of hacky fix to pass test_recycling_after_IOError
            self.frames = np.inf*np.ones((pix_r, pix_c, self.buffer_length), 
                                         np.float32)
            self.frames_gpu = cuda.mem_alloc(self.frames.size * self.frames.dtype.itemsize)
            cuda.memcpy_htod_async(self.frames_gpu, self.frames, 
                                   stream=self.bg_streamer)  # TODO - remove, part of hacky fix to pass test_recycling_after_IOError

    def _get_compiled_modules(self):
        """

        Get CUDA functions from PyCUDA SourceModule

        """
        self.nth_value_by_pixel = COMPILED_MODULE.get_function('nth_value_by_pixel')
        self.nth_value_by_pixel_shared_quicksort = COMPILED_MODULE.get_function('nth_value_by_pixel_shared_quicksort')
        self.nth_value_by_pixel_search_sort = COMPILED_MODULE.get_function('nth_value_by_pixel_search_sort')
        self.nth_value_by_pixel_search_sort_dynamic = COMPILED_MODULE.get_function('nth_value_by_pixel_search_sort_dynamic')
        self.clear_frame = COMPILED_MODULE.get_function('clear_frame')
        self.update_frame_and_convert_adu_to_e = COMPILED_MODULE.get_function('update_frame_and_convert_raw_adu_to_electrons')

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
            frame data as numpy.float32, contiguous in memory, units of ADU
        Returns
        -------
        Nothing

        """
        # send new frame to GPU
        cuda.memcpy_htod_async(self.new_frame_gpu, frame_data, stream=self.bg_streamer)
        # update the frame buffer on the GPU
        self.update_frame_and_convert_adu_to_e(self.frames_gpu, self.new_frame_gpu, position,
                                               np.int32(self.buffer_length), self.darkmap_gpu, self.flatmap_gpu,
                                               self.electrons_per_count,
                                               block=(self.slice_shape[0], 1, 1), grid=(self.slice_shape[1], 1),
                                               stream=self.bg_streamer)

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

        # make sure we have all the current frames on the CPU
        for frame in fresh:
            frame_data = np.ascontiguousarray(self.data_buffer.getSlice(frame), dtype=np.float32)
            # if we have unwanted frames on the GPU, replace them
            if len(uncleared) > 0:
                # find where to put new frame, and clean up the position dictionary in the process
                position = self.cur_positions.pop(uncleared.pop())

            else:  # need to add data without replacing anything
                # start from the front to make indexing on the GPU easier for never-before filled buffers. This lets us
                # initialize them as empty rather than filling them with inf the first time through
                position = np.int32(self.available.pop(0))

            self.update(frame, position, frame_data)

        # clear unwanted frames still living on the GPU
        for frame in uncleared:
            self.clear(frame)

    def calc_background(self, bg_indices):
        """

        Just calculates the background, does not return it, nor does it wait for the calculation to terminate before
        returning. This allows us to calculate the background asynchronously and later call Buffer.sync_calculation() to
        make sure everything is finished before using Buffer.cur_bg_gpu or pulling it back to the CPU.

        Parameters
        ----------
        bg_indices : iterable
            Frame indices to use when estimating the background for the frame about to be fit
        convert_to_electrons: bool
            Flag to camera-correct background and convert units from [ADU] to [e-]

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
                                                grid=(int(self.slice_shape[0] / 2), self.slice_shape[1]),
                                                stream=self.bg_streamer)
        elif full:
            # need to dynamically allocate
            filled = np.int32(self.buffer_length)
            self.nth_value_by_pixel_search_sort_dynamic(self.frames_gpu, self.index_to_grab,
                                                        filled, filled, self.cur_bg_gpu,
                                                        block=(self.buffer_length, 2, 1),
                                                        grid=(int(self.slice_shape[0] / 2), self.slice_shape[1]),
                                                        stream=self.bg_streamer,
                                                        shared=2 * 2 * self.buffer_length * 4)
        else:
            # buffer is not full, need to change which index we grab and dynamically allocate shared
            index_to_grab = np.int32(max([round(self.percentile * num_indices) - 1, 0]))
            # TODO - figure out another way to pass test_recycling_after_IOError unit test other than using full block to search partially filled buffer
            # filled = np.int32(num_indices)
            # self.nth_value_by_pixel_search_sort_dynamic(self.frames_gpu, index_to_grab,
            #                                             filled, np.int32(self.buffer_length), self.cur_bg_gpu,
            #                                             block=(num_indices, 2, 1),
            #                                             grid=(self.slice_shape[0] / 2, self.slice_shape[1]),
            #                                             stream=self.bg_streamer,
            #                                             shared=2 * 2 * num_indices * 4)
            # float * frames, const int n, const int spots_filled, const int buffer_length, float * nth_values
            self.nth_value_by_pixel_search_sort_dynamic(self.frames_gpu, index_to_grab,
                                                        np.int32(self.buffer_length), np.int32(self.buffer_length),
                                                        self.cur_bg_gpu,
                                                        block=(self.buffer_length, 2, 1),
                                                        grid=(int(self.slice_shape[0] / 2), self.slice_shape[1]),
                                                        stream=self.bg_streamer,
                                                        shared=2 * 2 * self.buffer_length * 4)

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
        This function is useful if you need the background estimation performed in a synchronous fashion.

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
        self.calc_background(bg_indices)
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
