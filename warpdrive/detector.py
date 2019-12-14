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

        self.dfilterBig = norm_uniform_filter(large_filter_size)
        self.dfilterSmall = norm_uniform_filter(small_filter_size)
        self.halfFiltBig = np.int32(0.5*large_filter_size)
        self.halfFiltSmall = np.int32(0.5*small_filter_size)
        self.guess_psf_sigma = np.float32(guess_psf_sigma)

        ###################### initialize PyCUDA ######################
        # select the first device and run in the default context
        self.dev = cuda.Device(0)

        self.dstreamer1 = cuda.Stream()
        self.dstreamer2 = cuda.Stream()
        self.vstreamer1 = cuda.Stream()
        self.vstreamer2 = cuda.Stream()

        self.streams = [self.dstreamer1, self.dstreamer2, self.vstreamer1, self.vstreamer2]
        self.num_streams = len(self.streams)

        ###################### Compile CUDA code ######################
        self._prepare_mod = source_prepare.prepare()
        self.prep_variance_over_gain_squared = self._prepare_mod.get_function('variance_over_gain_squared')
        self.raw_adu_to_e_and_estimate_noise = self._prepare_mod.get_function('raw_adu_to_e_and_estimate_noise')
        # compile difference-of-gaussian filters
        self.dog_mod = detectorCompileNBlock_sCMOS()
        self.dog_row_variance = self.dog_mod.get_function("dog_row_variance_convolution")
        self.dog_row = self.dog_mod.get_function("dog_row_convolution")
        self.dog_column = self.dog_mod.get_function("dog_column_convolution")
        self.weighted_dog = self.dog_mod.get_function("weighted_difference_of_gaussian_subtraction")

        # compile finding code
        self.findmod = finderCompile()
        self.maxfrow = self.findmod.get_function("maxfRowGPU")
        self.maxfcol = self.findmod.get_function("maxfColGPU")
        self.findPeaks = self.findmod.get_function("findPeaks")
        self.find_candidates_noise_thresh = self.findmod.get_function('find_candidates_noise_thresh')


        # compile fit
        self.fitmod = gaussMLE_Fang_David()
        # self.gaussAstig = self.fitmod.get_function("kernel_MLEFit_pix_threads_astig")
        # self.gaussAstigBkgndSub = self.fitmod.get_function("kernel_MLEFit_pix_threads_astig_subBkgnd")
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

    def allocateMem(self, dshape, ditemsize):
        """
        Allocate memory on the GPU. These allocations will be held until the detector object is destroyed.
        """

        self.dshape = [int(d) for d in dshape]  # use numpy int type for sizes; avoid potential np.int64
        self.dsize = dshape[0] * dshape[1] * int(ditemsize)

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
        self.filter1_gpu = cuda.mem_alloc(self.dfilterBig.size*self.dfilterBig.dtype.itemsize)  # cuda.mem_alloc(dfilter1.size * dfilter1.dtype.itemsize)
        self.filter2_gpu = cuda.mem_alloc(self.dfilterSmall.size*self.dfilterSmall.dtype.itemsize)  # cuda.mem_alloc(dfilter2.size * dfilter2.dtype.itemsize)
        self.maxfData_gpu = cuda.mem_alloc(self.dsize)

        self.candCount = np.array(0, dtype=np.int32)
        self.candCountZ = np.array(0, dtype=np.int32)
        self.candCount_gpu = cuda.mem_alloc(4)
        cuda.memcpy_htod(self.candCount_gpu, self.candCount)
        self.maxCandCount = np.int32(800)
        self.fitChunkSize = 32
        self.candPos = np.zeros(self.maxCandCount, dtype=np.int32) # This size of this array sets the limit on the number of candidate molecules per frame
        self.candPos_gpu = cuda.mem_alloc(self.candPos.size*self.candPos.dtype.itemsize)
        cuda.memcpy_htod(self.candPos_gpu, self.candPos)

        self.darkmap_gpu = cuda.mem_alloc(self.dsize)
        self.flatmap_gpu = cuda.mem_alloc(self.dsize)
        self.varmap_gpu = cuda.mem_alloc(self.dsize)
        self.noise_sigma_gpu = cuda.mem_alloc(self.dsize)
        self.variance_over_gain_squared_gpu = cuda.mem_alloc(self.dsize)


        #FIXME: not calculating CRLB (i.e. calcCRLB=0) causes an error
        self.calcCRLB = np.int32(1)


        self.dparsZ = np.zeros(6*self.fitChunkSize, dtype=np.float32)
        self.dpars = np.zeros(6*self.maxCandCount, dtype=np.float32)
        self.dpars_gpu = cuda.mem_alloc(self.dparsZ.size*self.dparsZ.dtype.itemsize)


        #self.CRLBs = np.zeros((6, self.candCount), dtype=np.float32)
        self.CRLBZ = np.zeros(6*self.fitChunkSize, dtype=np.float32)
        self.CRLB = np.zeros(6*self.maxCandCount, dtype=np.float32)
        self.CRLB_gpu = cuda.mem_alloc(self.CRLBZ.size*self.CRLBZ.dtype.itemsize)

        self.LLHZ = np.zeros(self.fitChunkSize, dtype=np.float32)
        self.LLH = np.zeros(self.maxCandCount, dtype=np.float32)
        self.LLH_gpu = cuda.mem_alloc(self.LLHZ.size*self.LLHZ.dtype.itemsize)

        self.dummyPosChunk = np.zeros(self.fitChunkSize, dtype=np.int32)
        self.candPosChunk_gpu = cuda.mem_alloc(self.dummyPosChunk.size*self.dummyPosChunk.dtype.itemsize)

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
                               stream=self.vstreamer1)
        cuda.memcpy_htod_async(self.varmap_gpu, np.ascontiguousarray(self.varmap, dtype=np.float32),
                               stream=self.vstreamer1)
        cuda.memcpy_htod_async(self.darkmap_gpu, np.ascontiguousarray(darkmap, dtype=np.float32),
                               stream=self.vstreamer1)

        # precalculate a per-pixel constant we'll use in the fit later. OK to leave this unsynced until later
        self.prep_variance_over_gain_squared(self.varmap_gpu, self.flatmap_gpu, np.float32(electrons_per_count),
                                             self.variance_over_gain_squared_gpu, block=(self.nrows, 1, 1),
                                             grid=(self.ncolumns, 1), stream=self.vstreamer2)

        # send our filters to device
        cuda.memcpy_htod_async(self.filter1_gpu, self.dfilterBig, stream=self.vstreamer1)
        cuda.memcpy_htod_async(self.filter2_gpu, self.dfilterSmall, stream=self.vstreamer1)

        # make sure all maps are on the gpu before splitting into multiple streams
        self.vstreamer1.synchronize()

        # Take row convolutions
        self.dog_row_variance(self.varmap_gpu, self.unif1v_gpu, self.filter1_gpu,
                     self.halfFiltBig, self.n_columns, block=(self.ncolumns, 1, 1),
                     grid=(self.nrows, 1), stream=self.vstreamer1)

        self.dog_row_variance(self.varmap_gpu, self.unif2v_gpu, self.filter2_gpu,
                     self.halfFiltSmall, self.n_columns, block=(self.ncolumns, 1, 1),
                     grid=(self.nrows, 1), stream=self.vstreamer2)

        # Take column convolutions
        self.dog_column(self.unif1v_gpu, self.filter1_gpu, self.n_rows, self.n_columns, self.halfFiltBig,
                   block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.vstreamer1)
        self.dog_column(self.unif2v_gpu, self.filter2_gpu, self.n_rows, self.n_columns, self.halfFiltSmall,
                   block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.vstreamer2)

        # make sure we're done before returning
        self.vstreamer1.synchronize()
        self.vstreamer2.synchronize()

        #cuda.memcpy_dtoh(self.dtarget, self.unif1v_gpu)
        #plt.show(plt.imshow(self.dtarget, interpolation='nearest'))

    def prepare_frame(self, data):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        cuda.memcpy_htod_async(self.data_gpu, self.data, stream=self.dstreamer1)

        self.raw_adu_to_e_and_estimate_noise(self.data_gpu, self.varmap_gpu, self.darkmap_gpu, self.flatmap_gpu,
                                             self._noise_factor, self._electrons_per_count, self._em_gain,
                                             self.noise_sigma_gpu,
                                             block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.dstreamer1)

        # dstreamer1 is synchronized in difference_of_gaussian_filter, so no need to repeat that here.


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
        # cuda.memcpy_htod_async(self.data_gpu, self.data, stream=self.dstreamer1)
        if background is None:
            # note that background array of zeros has already been sent to the GPU in allocateMem()

            # assign fit function for non-bkgnd subtraction case
            self.fitFunc = self.gaussAstig
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
                                       stream=self.dstreamer1)

                # assign our fit function
            self.fitFunc = self.pix_threads_astig_bkgndsub_mle

        # make sure self.prepare_frame() is finished, and if applicable, CPU background is on the GPU
        self.dstreamer1.synchronize()
        ############################# row convolutions ###################################
        self.dog_row(self.data_gpu, self.varmap_gpu, self.unif1_gpu, self.filter1_gpu,
                   self.halfFiltBig, self.bkgnd_gpu, block=(self.ncolumns, 1, 1),
                   grid=(self.nrows, 1), stream=self.dstreamer1)

        self.dog_row(self.data_gpu, self.varmap_gpu, self.unif2_gpu, self.filter2_gpu,
                   self.halfFiltSmall, self.bkgnd_gpu, block=(self.ncolumns, 1, 1),
                   grid=(self.nrows, 1), stream=self.dstreamer2)

        ############################# column convolutions ###################################
        self.dog_column(self.unif1_gpu, self.filter1_gpu, self.n_rows, self.n_columns, self.halfFiltBig,
                   block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.dstreamer1)
        self.dog_column(self.unif2_gpu, self.filter2_gpu, self.n_rows, self.n_columns, self.halfFiltSmall,
                   block=(self.nrows, 1, 1), grid=(self.ncolumns, 1), stream=self.dstreamer2)

        # dstreamer1 does not need to be synced because next call is in that stream.
        self.dstreamer2.synchronize()

        ############################# generate and subtract smooth iamges ###################################
        # (float *unifsmalldat,  float *unifsmallvar, float *uniflargedat, float *uniflargevar,int colsize, int halfFilt)
        self.weighted_dog(self.unif1_gpu, self.unif1v_gpu, self.unif2_gpu, self.unif2v_gpu, self.n_columns,
                      self.halfFiltBig, block=(self.ncolumns, 1, 1), grid=(self.nrows, 1), stream=self.dstreamer1)

        # A stream.sync is unnecessary here because the next call, maxfrow in getCand is also in dstreamer1


    def get_candidates(self, thresh=4, ROISize=16):
        """
        getCand should only be called after smoothFrame. It performs a maximum filter on the smoothed image, then finds
        all points (farther than half-ROIsize away from the frame-border) at which the maximum filter is equal to the
        smoothed image. The positions are then added to self.candPos, as measured in pixels taking the frame as a
        1D-array
        """
        # maxFilter size will be 2*halfMaxFilt + 1
        self.halfMaxFilt = np.int32(np.floor(0.5*ROISize) - 1)

        # fixme - make sure units on findpeaks.cu are all set once filter.cu is consistent
        # take maximum filter
        self.maxfrow(self.unif1_gpu, self.maxfData_gpu, self.n_columns, self.halfMaxFilt, block=(self.ncolumns, 1, 1),
                     grid=(self.nrows, 1), stream=self.dstreamer1)

        self.maxfcol(self.maxfData_gpu, self.n_columns, self.halfMaxFilt, block=(self.nrows, 1, 1),
                     grid=(self.ncolumns, 1), stream=self.dstreamer1)

        # candPos should be rezero'd at the end of fitting
        # FIXME: Check to see if removing the next line broke anything
        cuda.memcpy_htod_async(self.candCount_gpu, self.candCountZ, stream=self.dstreamer1)  #rezero the candidate count

        # determine whether to use simple threshold or pixel-dependent signal to noise threshold:
        # if noiseSig is None:
        #     findFunc = self.findPeaks
        # else:
            # cuda.memcpy_htod_async(self.noise_sigma_gpu, np.ascontiguousarray(noiseSig.squeeze(), dtype=np.float32),
            #                        stream=self.dstreamer1)
            # findFunc = self.findPeaksSNThresh


        self.find_candidates_noise_thresh(self.unif1_gpu, self.maxfData_gpu, np.float32(thresh), self.candCount_gpu,
                                          self.candPos_gpu, self.maxCandCount, np.int32(0.5*ROISize),
                                          self.noise_sigma_gpu,
                                          block=(self.ncolumns, 1, 1), grid=(self.nrows, 1), stream=self.dstreamer1)

        # retrieve number of candidates for block/grid allocation in fitting
        cuda.memcpy_dtoh_async(self.candCount, self.candCount_gpu, stream=self.dstreamer1)


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
        cuda.memcpy_dtoh_async(self.candPos, self.candPos_gpu, stream=self.dstreamer1)
        self.dstreamer1.synchronize()
        # fixme - make this work with unit-simplified and var/gain^2 pre-calc mle
        stream_counter = 0
        indy = 0
        while indy < self.candCount:
            # select stream
            to_use = self.streams[stream_counter % self.num_streams]
            # Re-zero fit outputs
            cuda.memcpy_htod_async(self.dpars_gpu, self.dparsZ, stream=to_use)
            cuda.memcpy_htod_async(self.CRLB_gpu, self.CRLBZ, stream=to_use)
            cuda.memcpy_htod_async(self.LLH_gpu, self.LLHZ, stream=to_use)

            numBlock = int(np.min([self.fitChunkSize, self.candCount - indy]))

            cuda.memcpy_htod_async(self.candPosChunk_gpu, self.candPos[indy:(indy+numBlock)], stream=to_use)

            # note that which fitFunc we use has already been decided by whether background was subtracted in detection
            self.fitFunc(self.data_gpu, self.guess_psf_sigma, self.iterations, self.dpars_gpu, self.CRLB_gpu,
                         self.LLH_gpu, self.variance_over_gain_squared_gpu, self.calcCRLB, self.candPosChunk_gpu,
                         self.n_columns, self.bkgnd_gpu,  # self.testROI_gpu,
                         block=(ROISize, ROISize, 1), grid=(numBlock, 1), stream=to_use)


            cuda.memcpy_dtoh_async(self.dpars[6*indy:6*(indy + numBlock)], self.dpars_gpu, stream=to_use)
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
        for stream_index in range(min([int((self.candCount - 1) / self.fitChunkSize) + 1, self.num_streams])):
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
        self.candPos[:n_cands] = cands
        cuda.memcpy_htod(self.candPos_gpu, self.candPos)
        self.candCount = np.int32(n_cands)

    def insertData(self, data, background=None):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        cuda.memcpy_htod(self.data_gpu, data)
        if background is not None:
            cuda.memcpy_htod(self.bkgnd_gpu, np.ascontiguousarray(background, dtype=np.float32))