"""
The detector class handles all of the CUDA memory allocations. It also compiles and holds handles to all of the
CUDA functions, which it has methods to execute.

Andrew Barentine - Spring 2016
"""

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import numpy as np
from .detector_cu import *
from . import source_prepare

def norm_uniform_filter(length):
    """
    Create a normalized uniform filter to use on the GPU. Note that our real-space filtering currently requires even
    filter sizes. To accommodate odd sizes, we use an even sized filter with the 0th element set to zero. Sub-pixel
    shifts in candidate emitter ROI detection resulting from use of even filters, or combinations of even and odd
    filters, are ignored.

    Parameters
    ----------
    length: int
        Length of uniform filter to return

    Returns
    -------
    filter: np.array
        1d normalized uniform filter

    """
    norm = 1./length
    if length % 2 == 0:  # even
        filter = np.ascontiguousarray(norm * np.ones(length), dtype=np.float32)
    else:  # odd
        # set up filter for use in same even-filter-sized convolution functions by setting first element 0
        filter = np.ascontiguousarray(norm * np.ones(length + 1), dtype=np.float32)
        filter[0] = 0
    return filter



class detector(object):
    def __init__(self, small_filter_size=4, large_filter_size=8, guess_psf_sigma=1.4):
        """
        Initialize PyCUDA and compile CUDA functions. All CUDA functions will be run in the default context
        in several streams initialized here.
        """
        self.iterations = np.int32(200)

        self.unifilt_large = norm_uniform_filter(large_filter_size)
        self.unifilt_small = norm_uniform_filter(small_filter_size)
        self.halfsize_large_filter = np.int32(0.5 * large_filter_size)
        self.halfsize_small_filter = np.int32(0.5 * small_filter_size)
        self.guess_psf_sigma = np.float32(guess_psf_sigma)

        ###################### initialize PyCUDA ######################
        # select the first device and run in the default context
        self.dev = cuda.Device(0)

        self.main_stream_r = cuda.Stream()
        self.main_stream_c = cuda.Stream()
        self.var_stream_r = cuda.Stream()
        self.var_stream_c = cuda.Stream()

        self.streams = [self.main_stream_r, self.main_stream_c, self.var_stream_r, self.var_stream_c]
        self.num_streams = len(self.streams)

        ###################### Compile CUDA code ######################
        self._prepare_mod = source_prepare.prepare()
        self.prep_variance_over_gain_squared = self._prepare_mod.get_function('variance_over_gain_squared')
        self.raw_adu_to_e_and_estimate_noise = self._prepare_mod.get_function('raw_adu_to_e_and_estimate_noise')
        # compile difference-of-gaussian filters
        self.dog_mod = compile_filters()
        self.dog_row_variance = self.dog_mod.get_function("dog_row_variance_convolution")
        self.dog_row = self.dog_mod.get_function("dog_row_convolution")
        self.dog_column = self.dog_mod.get_function("dog_column_convolution")
        self.weighted_dog = self.dog_mod.get_function("weighted_difference_of_gaussian_subtraction")

        # compile finding code
        self.findmod = compile_find_peaks()
        self.maxfrow = self.findmod.get_function("maxfRowGPU")
        self.maxfcol = self.findmod.get_function("maxfColGPU")
        self.find_peaks = self.findmod.get_function("findPeaks")
        self.find_candidates_noise_thresh = self.findmod.get_function('find_candidates_noise_thresh')


        # compile fit
        self.fitmod = compile_gauss_mle()
        # self.gaussAstig = self.fitmod.get_function("kernel_MLEFit_pix_threads_astig")
        self.pix_threads_astig_bkgndsub_mle = self.fitmod.get_function('pix_threads_astig_bkgndsub_mle')

        # print information about selected GPU
        print('Name: %s' % self.dev.name())
        print('Compute capability: %s' % (self.dev.compute_capability(),))
        info = pycuda.tools.DeviceData()
        print('Shared Memory size: %d' % info.shared_memory)
        print('Blocks per MP: %d' % info.thread_blocks_per_mp)
        print('MP count: %d' % self.dev.multiprocessor_count)
        #sharedinfo = self.context.get_shared_config()
        #print sharedinfo

    def allocate_memory(self, dshape):
        """
        Allocate memory on the GPU. These allocations will be held until the detector object is destroyed.
        """

        self.dshape = [int(d) for d in dshape]  # use numpy int type for sizes; avoid potential np.int64
        self.dsize = dshape[0] * dshape[1] * int(np.float32(0).itemsize)

        self.n_rows = np.int32(self.dshape[0])  # integers must be passed to PyCUDA functions as int32
        self.n_columns = np.int32(self.dshape[1])
        self.nrows = int(self.n_rows)  # integers used for block/grid allocations must be python int
        self.ncolumns = int(self.n_columns)


        ###################### Allocate resources on GPU ######################

        self.data_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
        self.bkgnd_gpu = cuda.mem_alloc(self.dsize)
        # fill background with zeros on gpu in case fitting without background subtraction is called
        cuda.memcpy_htod(self.bkgnd_gpu, np.ascontiguousarray(np.zeros(self.dshape), dtype=np.float32))
        # self.noiseSigma_gpu = cuda.mem_alloc(self.dsize)
        self.unif1_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
        self.unif2_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
        self.unif1v_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
        self.unif2v_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
        self.invvar_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(varmap.size * varmap.dtype.itemsize)
        self.filter1_gpu = cuda.mem_alloc(self.unifilt_large.size * self.unifilt_large.dtype.itemsize)  # cuda.mem_alloc(dfilter1.size * dfilter1.dtype.itemsize)
        self.filter2_gpu = cuda.mem_alloc(self.unifilt_small.size * self.unifilt_small.dtype.itemsize)  # cuda.mem_alloc(dfilter2.size * dfilter2.dtype.itemsize)
        self.maxf_data_gpu = cuda.mem_alloc(self.dsize)

        self.n_candidates = np.array(0, dtype=np.int32)
        self.n_candidate_zeroer = np.array(0, dtype=np.int32)
        self.n_candidates_gpu = cuda.mem_alloc(4)
        cuda.memcpy_htod(self.n_candidates_gpu, self.n_candidates)
        self.n_max_candidates_per_frame = np.int32(800)
        self.fit_chunk_size = 32
        self.candidate_positions = np.zeros(self.n_max_candidates_per_frame, dtype=np.int32) # This size of this array sets the limit on the number of candidate molecules per frame
        self.candidate_positions_gpu = cuda.mem_alloc(self.candidate_positions.size * self.candidate_positions.dtype.itemsize)
        cuda.memcpy_htod(self.candidate_positions_gpu, self.candidate_positions)

        self.darkmap_gpu = cuda.mem_alloc(self.dsize)
        self.flatmap_gpu = cuda.mem_alloc(self.dsize)
        self.varmap_gpu = cuda.mem_alloc(self.dsize)
        self.noise_sigma_gpu = cuda.mem_alloc(self.dsize)
        self.variance_over_gain_squared_gpu = cuda.mem_alloc(self.dsize)


        #FIXME: not calculating CRLB (i.e. calcCRLB=0) causes an error
        self.calculate_crb = np.int32(1)


        self.fit_res_zeroer = np.zeros(6 * self.fit_chunk_size, dtype=np.float32)
        self.fit_res = np.zeros(6 * self.n_max_candidates_per_frame, dtype=np.float32)
        self.fit_res_gpu = cuda.mem_alloc(self.fit_res_zeroer.size * self.fit_res_zeroer.dtype.itemsize)


        #self.CRLBs = np.zeros((6, self.n_candidates), dtype=np.float32)
        self.CRLB_zeroer = np.zeros(6 * self.fit_chunk_size, dtype=np.float32)
        self.CRLB = np.zeros(6 * self.n_max_candidates_per_frame, dtype=np.float32)
        self.CRLB_gpu = cuda.mem_alloc(self.CRLB_zeroer.size * self.CRLB_zeroer.dtype.itemsize)

        self.LLH_zeroer = np.zeros(self.fit_chunk_size, dtype=np.float32)
        self.LLH = np.zeros(self.n_max_candidates_per_frame, dtype=np.float32)
        self.LLH_gpu = cuda.mem_alloc(self.LLH_zeroer.size * self.LLH_zeroer.dtype.itemsize)

        dummy_position_chunk = np.zeros(self.fit_chunk_size, dtype=np.int32)
        self.candidate_position_chunk_gpu = cuda.mem_alloc(dummy_position_chunk.size * dummy_position_chunk.dtype.itemsize)

        # for troubleshooting:
        self.dtarget = np.zeros(self.dshape, dtype=np.float32)

    def prepare_maps(self, darkmap, varmap, flatmap, electrons_per_count, noise_factor, em_gain):
        """
        sends the variance and gain maps to the GPU, where they will remain. This function must be called before
        smoothFrame. When the FOV is shifted, this must be called again in order to update the camera maps held by the
        GPU.

        Parameters
        ----------
            varmap: ndarray
                variance map [e-^2]
            flatmap: ndarray
                flatmap [unitless, mean-normalized], can be calculated as (1/gain)/mean(1/gain). The conversion from
                flatmap to gainmap [ADU/e-] is done in-line as it is sent to GPU.
            electronsPerCount: float
                Conversion factor from ADU to e-, [e-/ADU]

        Notes
        -----
        varmap was previously in units of [ADU^2] but changed 2019/11/14 to make consistent with PYME and support direct
        use of PYME-generated variance maps. Older [ADU^2] camera maps will need to be resaved as [e-^2] as they are no
        longer compatible.
        """

        # store varmap as an attribute so the fit factory can check if the camera ROI has shifted
        self.varmap = varmap

        # store camera noise properties as attributes for use in estimating noise standard deviation
        self._electrons_per_count = np.float32(electrons_per_count)
        self._noise_factor = np.float32(noise_factor)
        self._em_gain = np.float32(em_gain)

        # send maps to the gpu
        cuda.memcpy_htod_async(self.flatmap_gpu, np.ascontiguousarray(flatmap, dtype=np.float32),
                               stream=self.var_stream_r)
        cuda.memcpy_htod_async(self.varmap_gpu, np.ascontiguousarray(self.varmap, dtype=np.float32),
                               stream=self.var_stream_r)
        cuda.memcpy_htod_async(self.darkmap_gpu, np.ascontiguousarray(darkmap, dtype=np.float32),
                               stream=self.var_stream_r)

        # precalculate a per-pixel constant we'll use in the fit later. OK to leave this unsynced until later
        self.prep_variance_over_gain_squared(self.varmap_gpu, self.flatmap_gpu, np.float32(electrons_per_count),
                                             self.variance_over_gain_squared_gpu, block=(self.nrows, 1, 1),
                                             grid=(self.ncolumns, 1), stream=self.var_stream_c)

        # send our filters to device
        cuda.memcpy_htod_async(self.filter1_gpu, self.unifilt_large, stream=self.var_stream_r)
        cuda.memcpy_htod_async(self.filter2_gpu, self.unifilt_small, stream=self.var_stream_r)

        # make sure all maps are on the gpu before splitting into multiple streams
        self.var_stream_r.synchronize()

        # Take row convolutions
        self.dog_row_variance(self.varmap_gpu, self.unif1v_gpu, self.filter1_gpu,
                              self.halfsize_large_filter, self.n_columns, block=(self.ncolumns, 1, 1),
                              grid=(self.nrows, 1), stream=self.var_stream_r)

        self.dog_row_variance(self.varmap_gpu, self.unif2v_gpu, self.filter2_gpu,
                              self.halfsize_small_filter, self.n_columns, block=(self.ncolumns, 1, 1),
                              grid=(self.nrows, 1), stream=self.var_stream_c)

        # Take column convolutions
        self.dog_column(self.unif1v_gpu, self.filter1_gpu, self.n_rows, self.n_columns, self.halfsize_large_filter,
                        block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.var_stream_r)
        self.dog_column(self.unif2v_gpu, self.filter2_gpu, self.n_rows, self.n_columns, self.halfsize_small_filter,
                        block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.var_stream_c)

        # make sure we're done before returning
        self.var_stream_r.synchronize()
        self.var_stream_c.synchronize()

        #cuda.memcpy_dtoh(self.dtarget, self.unif1v_gpu)
        #plt.show(plt.imshow(self.dtarget, interpolation='nearest'))

    def prepare_frame(self, data):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        cuda.memcpy_htod_async(self.data_gpu, self.data, stream=self.main_stream_r)

        self.raw_adu_to_e_and_estimate_noise(self.data_gpu, self.varmap_gpu, self.darkmap_gpu, self.flatmap_gpu,
                                             self._noise_factor, self._electrons_per_count, self._em_gain,
                                             self.noise_sigma_gpu,
                                             block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.main_stream_r)

        # main_stream_r is synchronized in difference_of_gaussian_filter, so no need to repeat that here.


    def difference_of_gaussian_filter(self, background=None):
        """
        smoothFrame passes a single frame of data to the GPU, and performs two 2D convolutions with different kernel
        sizes before subtracting the two. This is done in a variance-weighted fashion. For description, see supplemental
        materials of 10.1038/nmeth.2488.

        Args:
            photondat: [ADU]
            bkgnd: [ADU]

        Returns:
            nothing, but fit parameters are held (on both CPU and GPU) by detector instance

        """
        #print('Data: mu = %f +- %f' % (np.mean(photondat), np.std(photondat)))
        #print('Background: %s' % (bkgnd is None))
        # make sure that the data is contiguous, and send to GPU
        # self.data = np.ascontiguousarray(data, dtype=np.float32)
        # cuda.memcpy_htod_async(self.data_gpu, self.data, stream=self.main_stream_r)
        if background is None:
            # make sure background is zero'd on gpu
            cuda.memcpy_htod_async(self.bkgnd_gpu, np.ascontiguousarray(np.zeros(self.dshape), dtype=np.float32),
                                   stream=self.main_stream_r)
        else:  # background is either already on the GPU or was passed to this function
            try:
                # if we have the gpu buffer, pass off the pointer to the device memory
                self.bkgnd_gpu = background.cur_bg_gpu
                # make sure the calculation is finished
                background.sync_calculation()
            except AttributeError:
                # background should be array, pass it to the GPU now. FIXME - make sure background is in [e-] here
                # send bkgnd via stream 1 because in current implementation, bkgnd is needed in row convolution
                cuda.memcpy_htod_async(self.bkgnd_gpu, np.ascontiguousarray(background, dtype=np.float32),
                                       stream=self.main_stream_r)

                # assign our fit function
            # self.fitFunc = self.pix_threads_astig_bkgndsub_mle

        # make sure self.prepare_frame() is finished, and if applicable, CPU background is on the GPU
        self.main_stream_r.synchronize()
        ############################# row convolutions ###################################
        self.dog_row(self.data_gpu, self.varmap_gpu, self.unif1_gpu, self.filter1_gpu,
                     self.halfsize_large_filter, self.bkgnd_gpu, block=(self.ncolumns, 1, 1),
                     grid=(self.nrows, 1), stream=self.main_stream_r)

        self.dog_row(self.data_gpu, self.varmap_gpu, self.unif2_gpu, self.filter2_gpu,
                     self.halfsize_small_filter, self.bkgnd_gpu, block=(self.ncolumns, 1, 1),
                     grid=(self.nrows, 1), stream=self.main_stream_c)

        ############################# column convolutions ###################################
        self.dog_column(self.unif1_gpu, self.filter1_gpu, self.n_rows, self.n_columns, self.halfsize_large_filter,
                        block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.main_stream_r)
        self.dog_column(self.unif2_gpu, self.filter2_gpu, self.n_rows, self.n_columns, self.halfsize_small_filter,
                        block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.main_stream_c)

        # main_stream_r does not need to be synced because next call is in that stream.
        self.main_stream_c.synchronize()

        ############################# generate and subtract smooth iamges ###################################
        # (float *unifsmalldat,  float *unifsmallvar, float *uniflargedat, float *uniflargevar,int colsize, int halfFilt)
        self.weighted_dog(self.unif1_gpu, self.unif1v_gpu, self.unif2_gpu, self.unif2v_gpu, self.n_columns,
                          self.halfsize_large_filter, block=(self.ncolumns, 1, 1), grid=(self.nrows, 1), stream=self.main_stream_r)

        # A stream.sync is unnecessary here because the next call, maxfrow in getCand is also in main_stream_r


    def get_candidates(self, thresh=4, ROISize=16):
        """
        getCand should only be called after smoothFrame. It performs a maximum filter on the smoothed image, then finds
        all points (farther than half-ROIsize away from the frame-border) at which the maximum filter is equal to the
        smoothed image. The positions are then added to self.candidate_positions, as measured in pixels taking the frame as a
        1D-array
        """
        # maxFilter size will be 2*halfMaxFilt + 1
        self.halfMaxFilt = np.int32(np.floor(0.5*ROISize) - 1)

        # fixme - make sure units on findpeaks.cu are all set once filter.cu is consistent
        # take maximum filter
        self.maxfrow(self.unif1_gpu, self.maxf_data_gpu, self.n_columns, self.halfMaxFilt, block=(self.ncolumns, 1, 1),
                     grid=(self.nrows, 1), stream=self.main_stream_r)

        self.maxfcol(self.maxf_data_gpu, self.n_columns, self.halfMaxFilt, block=(self.nrows, 1, 1),
                     grid=(self.ncolumns, 1), stream=self.main_stream_r)

        # candidate_positions should be rezero'd at the end of fitting
        # FIXME: Check to see if removing the next line broke anything
        cuda.memcpy_htod_async(self.n_candidates_gpu, self.n_candidate_zeroer, stream=self.main_stream_r)  #rezero the candidate count

        # determine whether to use simple threshold or pixel-dependent signal to noise threshold:
        # if noiseSig is None:
        #     findFunc = self.find_peaks
        # else:
            # cuda.memcpy_htod_async(self.noise_sigma_gpu, np.ascontiguousarray(noiseSig.squeeze(), dtype=np.float32),
            #                        stream=self.main_stream_r)
            # findFunc = self.findPeaksSNThresh


        self.find_candidates_noise_thresh(self.unif1_gpu, self.maxf_data_gpu, np.float32(thresh), self.n_candidates_gpu,
                                          self.candidate_positions_gpu, self.n_max_candidates_per_frame, np.int32(0.5 * ROISize),
                                          self.noise_sigma_gpu,
                                          block=(self.ncolumns, 1, 1), grid=(self.nrows, 1), stream=self.main_stream_r)

        # retrieve number of candidates for block/grid allocation in fitting
        cuda.memcpy_dtoh_async(self.n_candidates, self.n_candidates_gpu, stream=self.main_stream_r)


    def fit_candidates(self, ROISize=16):
        """
        This function runs David Baddeley's pixel-wise GPU fit, and is pretty darn fast. The fit is an MLE fit, with a
        noise-model which accounts for sCMOS statistics, i.e. a gaussian random variable (read-noise) added to a Poisson
        -random variable. This model is described in 10.1038/nmeth.2488.

        To allow multiple processes to share the GPU, each proccess only fits 32 ROI's at a time (corresponding to 32
        candidate molecules)

        Args:
            ROISize: Integer size of (square) subROI to be fit

        Returns:
            nothing, but fit parameters are held (on both CPU and GPU) by detector instance

        """

        # Pull candidates back to host so we can insert them chunk by chunk into the fit
        # fixme - surely we can eliminate this back-and-forth transfer
        cuda.memcpy_dtoh_async(self.candidate_positions, self.candidate_positions_gpu, stream=self.main_stream_r)
        self.main_stream_r.synchronize()
        # fixme - make this work with unit-simplified and var/gain^2 pre-calc mle
        stream_counter = 0
        indy = 0
        while indy < self.n_candidates:
            # select stream
            to_use = self.streams[stream_counter % self.num_streams]
            # Re-zero fit outputs
            cuda.memcpy_htod_async(self.fit_res_gpu, self.fit_res_zeroer, stream=to_use)
            cuda.memcpy_htod_async(self.CRLB_gpu, self.CRLB_zeroer, stream=to_use)
            cuda.memcpy_htod_async(self.LLH_gpu, self.LLH_zeroer, stream=to_use)

            numBlock = int(np.min([self.fit_chunk_size, self.n_candidates - indy]))

            cuda.memcpy_htod_async(self.candidate_position_chunk_gpu, self.candidate_positions[indy:(indy + numBlock)], stream=to_use)

            # note that which fitFunc we use has already been decided by whether background was subtracted in detection
            self.pix_threads_astig_bkgndsub_mle(self.data_gpu, self.guess_psf_sigma, self.iterations, self.fit_res_gpu, self.CRLB_gpu,
                                                self.LLH_gpu, self.variance_over_gain_squared_gpu, self.calculate_crb, self.candidate_position_chunk_gpu,
                                                self.n_columns, self.bkgnd_gpu,  # self.testROI_gpu,
                                                block=(ROISize, ROISize, 1), grid=(numBlock, 1), stream=to_use)


            cuda.memcpy_dtoh_async(self.fit_res[6 * indy:6 * (indy + numBlock)], self.fit_res_gpu, stream=to_use)
            cuda.memcpy_dtoh_async(self.CRLB[6*indy:6*(indy + numBlock)], self.CRLB_gpu, stream=to_use)
            cuda.memcpy_dtoh_async(self.LLH[indy:(indy + numBlock)], self.LLH_gpu, stream=to_use)
            indy += numBlock
            stream_counter += 1


        # uncomment if using testROI for dummy-checking:
        #cuda.memcpy_dtoh(self.testROI, self.testROI_gpu)
        #import matplotlib.pyplot as plt
        #plt.imshow(self.testROI, interpolation='nearest')
        #plt.show()

        # synchronize streams, recall a fast version of ceil(a/b) = (a - 1)/b + 1
        for stream_index in range(min([int((self.n_candidates - 1) / self.fit_chunk_size) + 1, self.num_streams])):
            self.streams[stream_index].synchronize()

        return

    def offIt(self):
        """
        This function COULD be called to clean a detector object context off of the GPU gracefully, HOWEVER, it would
        also clean ALL detector objects off the GPU and kill the context. This function is depreciated, but left here
        so the world knows not to do it.
        """
        self.context.pop()
        return

    def insertTestCandidates(self, testCand):
        """
        This function is only for testing. Allowing you to specify the candidate molecule positions in a scope with
        cuda drivers present
        """
        cands = np.asarray(testCand)
        n_cands = len(cands)
        self.candidate_positions[:n_cands] = cands
        cuda.memcpy_htod(self.candidate_positions_gpu, self.candidate_positions)
        self.n_candidates = np.int32(n_cands)

    def insertData(self, data, background=None):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        cuda.memcpy_htod(self.data_gpu, data)
        if background is not None:
            cuda.memcpy_htod(self.bkgnd_gpu, np.ascontiguousarray(background, dtype=np.float32))
