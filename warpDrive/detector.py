"""
The detector class handles all of the CUDA memory allocations. It also compiles and holds handles to all of the
CUDA functions, which it has methods to execute.

Andrew Barentine - Spring 2016
"""

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import numpy as np
from detector_cu import *



class detector:

    def __init__(self, dshape, ditemsize, dfilterBig, dfilterSmall):
        """
        Initialize PyCUDA and compile CUDA functions. All CUDA functions will be run in the default context
        in several streams initialized here.
        """
        self.dshape = dshape
        self.dsize = dshape[0] * dshape[1] *ditemsize
        self.dfilterBig = dfilterBig
        self.dfilterSmall = dfilterSmall


        self.rowsize = np.int32(self.dshape[0])  # integers must be passed to PyCUDA functions as int32's
        self.colsize = np.int32(self.dshape[1])
        self.rsize = int(self.rowsize)  # need integer type for block/grid size definitions
        self.csize = int(self.colsize)

        self.halfFiltBig = np.int32(0.5*len(self.dfilterBig))
        self.halfFiltSmall = np.int32(0.5*len(self.dfilterSmall))

        ###################### initialize PyCUDA ######################
        # select the first device and run in the default context
        self.dev = cuda.Device(0)

        self.dstreamer1 = cuda.Stream()
        self.dstreamer2 = cuda.Stream()
        self.vstreamer1 = cuda.Stream()
        self.vstreamer2 = cuda.Stream()

        ###################### Compile CUDA code ######################
        # compile smoothing code
        self.smoothmod = detectorCompileNBlock_sCMOS()
        self.rfunc_v = self.smoothmod.get_function("convRowGPU_var")
        self.rfunc = self.smoothmod.get_function("convRowGPU")
        self.cfunc = self.smoothmod.get_function("convColGPU")
        self.smoothIm = self.smoothmod.get_function("smoothImGPU")

        # compile finding code
        self.findmod = finderCompile()
        self.maxfrow = self.findmod.get_function("maxfRowGPU")
        self.maxfcol = self.findmod.get_function("maxfColGPU")
        self.findPeaks = self.findmod.get_function("findPeaks")
        self.findPeaksSNThresh = self.findmod.get_function("findPeaksSNThresh")


        # compile fit
        self.fitmod = gaussMLE_Fang_David()
        self.gaussAstig = self.fitmod.get_function("kernel_MLEFit_pix_threads_astig")
        self.gaussAstigBkgndSub = self.fitmod.get_function("kernel_MLEFit_pix_threads_astig_subBkgnd")

        # print information about selected GPU
        print 'Name:', self.dev.name()
        print 'Compute capability:', self.dev.compute_capability()
        info = pycuda.tools.DeviceData()
        print 'Shared Memory size:', info.shared_memory
        print 'Blocks per MP:', info.thread_blocks_per_mp
        print 'MP count:', self.dev.multiprocessor_count
        #sharedinfo = self.context.get_shared_config()
        #print sharedinfo

    def allocateMem(self):
        """
        Allocate memory on the GPU. These allocations will be held until the detector object is destroyed.
        """

        ###################### Allocate resources on GPU ######################

        self.data_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
        self.bkgnd_gpu = cuda.mem_alloc(self.dsize)
        # fill background with zeros on gpu in case fitting without background subtraction is called
        cuda.memcpy_htod(self.bkgnd_gpu, np.ascontiguousarray(np.zeros(self.dshape), dtype=np.float32))
        self.noiseSigma_gpu = cuda.mem_alloc(self.dsize)
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


        # for troubleshooting:
        self.dtarget = np.zeros(self.dshape, dtype=np.float32)

        self.gain_gpu = cuda.mem_alloc(self.dsize)


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


    def prepvar(self, varmap, flatmap, electronsPerCount):
        """
        prepvar sends the variance and gain maps to the GPU, where they will remain. This function must be called before
        smoothFrame. When the FOV is shifted, this must be called again in order to update the camera maps held by the
        GPU.

        Args:
            varmap: variance map
            flatmap: flatmap, i.e. (1/gain)/<1/gain>. The conversion from flatmap to gainmap is done in-line as it is
                sent to GPU. Note that PYME-style flatmaps are also normalized to one, so use electronsPerCount to
                convert to units of [ADU/e-]
            electronsPerCount: [e-/ADU], i.e. <1/gain>

        Returns:
            nothing
        """
        print('Variance map: mu = %f +- %f' % (np.mean(varmap), np.std(varmap)))
        print('Flatfield map: mu = %f +- %f' % (np.mean(flatmap), np.std(flatmap)))
        print('ElectronsPerCount: %f' % electronsPerCount)
        self.varmap = varmap

        # note that PYME-style flatmaps are unitless, need to convert to gain in units of [ADU/e-]
        cuda.memcpy_htod_async(self.gain_gpu, np.ascontiguousarray(1./(electronsPerCount*flatmap), dtype=np.float32), stream=self.vstreamer1)

        cuda.memcpy_htod_async(self.filter1_gpu, self.dfilterBig, stream=self.dstreamer1)
        cuda.memcpy_htod_async(self.filter2_gpu, self.dfilterSmall, stream=self.dstreamer1)
        cuda.memcpy_htod_async(self.invvar_gpu, np.ascontiguousarray(self.varmap, dtype=np.float32), stream=self.dstreamer1)

        # Take row convolutions
        self.rfunc_v(self.invvar_gpu, self.unif1v_gpu, self.filter1_gpu,
                   self.halfFiltBig, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer1)

        self.rfunc_v(self.invvar_gpu, self.unif2v_gpu, self.filter2_gpu,
                   self.halfFiltSmall, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer2)

        # Take column convolutions
        self.cfunc(self.unif1v_gpu, self.filter1_gpu, self.rowsize, self.colsize, self.halfFiltBig,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.vstreamer1)
        self.cfunc(self.unif2v_gpu, self.filter2_gpu, self.rowsize, self.colsize, self.halfFiltSmall,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.vstreamer2)

        # Pause until complete
        self.vstreamer1.synchronize()
        self.vstreamer2.synchronize()

        #cuda.memcpy_dtoh(self.dtarget, self.unif1v_gpu)
        #plt.show(plt.imshow(self.dtarget, interpolation='nearest'))

    def smoothFrame(self, photondat, bkgnd=None):
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
        print('Data: mu = %f +- %f' % (np.mean(photondat), np.std(photondat)))
        print('Background: %s' % (bkgnd is None))
        # make sure that the data is contiguous, and send to GPU
        self.data = np.ascontiguousarray(photondat, dtype=np.float32)
        cuda.memcpy_htod_async(self.data_gpu, self.data, stream=self.dstreamer1)
        if bkgnd is None:
            # note that background array of zeros has already been sent to the GPU in allocateMem()

            # assign fit function for non-bkgnd subtraction case
            self.fitFunc = self.gaussAstig
        else:
            print('Background: mu = %f +- %f' % (np.mean(bkgnd), np.std(bkgnd)))
            # send bkgnd via stream 1 because in current implementation, bkgnd is needed in row convolution
            cuda.memcpy_htod_async(self.bkgnd_gpu, np.ascontiguousarray(bkgnd, dtype=np.float32),
                                   stream=self.dstreamer1)
            # assign our fit function
            self.fitFunc = self.gaussAstigBkgndSub

        # make sure data is on the GPU before taking convolutions (otherwise dstreamer2 could potentially fire too soon)
        self.dstreamer1.synchronize()

        ############################# row convolutions ###################################
        self.rfunc(self.data_gpu, self.invvar_gpu, self.unif1_gpu, self.gain_gpu, self.filter1_gpu,
                   self.halfFiltBig, self.colsize, self.bkgnd_gpu, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer1)

        self.rfunc(self.data_gpu, self.invvar_gpu, self.unif2_gpu, self.gain_gpu, self.filter2_gpu,
                   self.halfFiltSmall, self.colsize, self.bkgnd_gpu, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer2)

        ############################# column convolutions ###################################
        self.cfunc(self.unif1_gpu, self.filter1_gpu, self.rowsize, self.colsize, self.halfFiltBig,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.dstreamer1)
        self.cfunc(self.unif2_gpu, self.filter2_gpu, self.rowsize, self.colsize, self.halfFiltSmall,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.dstreamer2)

        # dstreamer1 does not need to be synced because next call is in that stream.
        self.dstreamer2.synchronize()

        ############################# generate and subtract smooth iamges ###################################
        # (float *unifsmalldat,  float *unifsmallvar, float *uniflargedat, float *uniflargevar,int colsize, int halfFilt)
        self.smoothIm(self.unif1_gpu, self.unif1v_gpu, self.unif2_gpu, self.unif2v_gpu, self.colsize,
                      self.halfFiltBig, block=(self.csize, 1, 1), grid=(self.rsize, 1), stream=self.dstreamer1)

        # A stream.sync is unnecessary here because the next call, maxfrow in getCand is also in dstreamer1


    def getCand(self, thresh=4, ROISize=16, noiseSig=None, ePerADU=1.0):
        """
        getCand should only be called after smoothFrame. It performs a maximum filter on the smoothed image, then finds
        all points (farther than half-ROIsize away from the frame-border) at which the maximum filter is equal to the
        smoothed image. The positions are then added to self.candPos, as measured in pixels taking the frame as a
        1D-array
        """
        # maxFilter size will be 2*halfMaxFilt + 1
        self.halfMaxFilt = np.int32(np.floor(0.5*ROISize) - 1)

        # take maximum filter
        self.maxfrow(self.unif1_gpu, self.maxfData_gpu, self.colsize, self.halfMaxFilt, block=(self.csize, 1, 1),
                     grid=(self.rsize, 1), stream=self.dstreamer1)

        self.maxfcol(self.maxfData_gpu, self.colsize, self.halfMaxFilt, block=(self.rsize, 1, 1),
                     grid=(self.csize, 1), stream=self.dstreamer1)

        # candPos should be rezero'd at the end of fitting
        # FIXME: Check to see if removing the next line broke anything
        cuda.memcpy_htod_async(self.candCount_gpu, self.candCountZ, stream=self.dstreamer1)  #rezero the candidate count

        # determine whether to use simple threshold or pixel-dependent signal to noise threshold:
        if noiseSig is None:
            findFunc = self.findPeaks
        else:
            cuda.memcpy_htod_async(self.noiseSigma_gpu, np.ascontiguousarray(noiseSig.squeeze(), dtype=np.float32),
                                   stream=self.dstreamer1)
            findFunc = self.findPeaksSNThresh

        # Check at which points the smoothed frame is equal to the maximum filter of the smooth frame
        findFunc(self.unif1_gpu, self.maxfData_gpu, np.float32(thresh), self.colsize, self.candCount_gpu,
                   self.candPos_gpu, np.int32(0.5*ROISize), self.maxCandCount, self.noiseSigma_gpu, np.float32(ePerADU),
                   block=(self.csize, 1, 1), grid=(self.rsize, 1), stream=self.dstreamer1)

        # retrieve number of candidates for block/grid allocation in fitting
        cuda.memcpy_dtoh_async(self.candCount, self.candCount_gpu, stream=self.dstreamer1)


    def fitItToWinIt(self, ROISize=16):
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
        cuda.memcpy_dtoh_async(self.candPos, self.candPos_gpu, stream=self.dstreamer1)

        indy = 0
        while indy < self.candCount:
            # Re-zero fit outputs
            cuda.memcpy_htod_async(self.dpars_gpu, self.dparsZ, stream=self.dstreamer1)
            cuda.memcpy_htod_async(self.CRLB_gpu, self.CRLBZ, stream=self.dstreamer1)
            cuda.memcpy_htod_async(self.LLH_gpu, self.LLHZ, stream=self.dstreamer1)

            numBlock = int(np.min([self.fitChunkSize, self.candCount - indy]))

            cuda.memcpy_htod_async(self.candPosChunk_gpu, self.candPos[indy:(indy+numBlock)], stream=self.dstreamer1)

            # note that which fitFunc we use has already been decided by whether background was subtracted in detection
            self.fitFunc(self.data_gpu, np.float32(1.4), np.int32(200),
                    self.dpars_gpu, self.CRLB_gpu, self.LLH_gpu, self.invvar_gpu, self.gain_gpu,
                    self.calcCRLB, self.candPosChunk_gpu, np.int32(self.csize), np.int32(0), self.bkgnd_gpu,  # self.testROI_gpu,
                    block=(ROISize, ROISize, 1), grid=(numBlock, 1), stream=self.dstreamer1)


            cuda.memcpy_dtoh_async(self.dpars[6*indy:6*(indy + numBlock)], self.dpars_gpu, stream=self.dstreamer1)
            cuda.memcpy_dtoh_async(self.CRLB[6*indy:6*(indy + numBlock)], self.CRLB_gpu, stream=self.dstreamer1)
            cuda.memcpy_dtoh_async(self.LLH[indy:(indy + numBlock)], self.LLH_gpu, stream=self.dstreamer1)
            indy += numBlock


        # uncomment if using testROI for dummy-checking:
        #cuda.memcpy_dtoh(self.testROI, self.testROI_gpu)
        #import matplotlib.pyplot as plt
        #plt.imshow(self.testROI, interpolation='nearest')
        #plt.show()

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
        cuda.memcpy_htod(self.candPos_gpu, np.ascontiguousarray(testCand, dtype=np.int32))
        self.candCount = np.int32(len([testCand]))
        return

    def insertData(self, data):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        cuda.memcpy_htod(self.data_gpu, data)
        return

def normUnifFilter(siz):
    return np.ascontiguousarray((1./siz)*np.ones(siz, dtype=np.float32), dtype=np.float32)
