import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import pycuda.tools
import numpy as np
from detector_cu import *


#import matplotlib.pyplot as plt
#from matplotlib import cm
#import time
#import scipy.io











class detector:

    def __init__(self, dshape, ditemsize, dfilterBig, dfilterSmall):
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

        #self.thresh = np.float32(4.0601)
        #self.thresh = np.float32(100)
        # FIXME: decide reasonable size for maximum filter
        #self.halfMaxFilt = np.int32(7)  # maxFilter size will be 2*halfMaxFilt + 1



        ###################### initialize PyCUDA ######################
        self.dev = cuda.Device(0)
        #self.context = self.dev.make_context()

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
        self.findpeaks = self.findmod.get_function("findPeaks")


        # compile fit
        self.fitmod = gaussMLE_Fang_David()
        self.gaussAstig = self.fitmod.get_function("kernel_MLEFit_pix_threads_astig")


        print 'Name:', self.dev.name()
        print 'Compute capability:', self.dev.compute_capability()
        info = pycuda.tools.DeviceData()
        print 'Shared Memory size:', info.shared_memory
        print 'Blocks per MP:', info.thread_blocks_per_mp
        #info2 = cuda.device_attribute(flags='MULTIPROCESSOR_COUNT')
        print 'MP count:', self.dev.multiprocessor_count
        #sharedinfo = self.context.get_shared_config()
        #print sharedinfo

    def allocateMem(self):
        ###################### Allocate resources on GPU ######################

        self.data_gpu = cuda.mem_alloc(self.dsize)  # cuda.mem_alloc(rawdat.size * rawdat.dtype.itemsize)
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
        #  cuda.memcpy_htod(self.candCount_gpu, self.candCountZ)
        self.candPos = np.zeros(400, dtype=np.int32) # This size of this array sets the limit on the number of candidate molecules per frame
        self.candPos_gpu = cuda.mem_alloc(self.candPos.size*self.candPos.dtype.itemsize)
        cuda.memcpy_htod(self.candPos_gpu, self.candPos)

        #current soln for leaving gain*var on GPU for smoothing:
        self.gtimesv_gpu = cuda.mem_alloc(self.dsize)

        # for troubleshooting:
        self.dtarget = np.zeros(self.dshape, dtype=np.float32)
        #self.dtarget2 = np.zeros(self.dshape, dtype=np.float32)




        #self.gain = np.ascontiguousarray(np.ones(self.dshape, dtype=np.float32))#np.ascontiguousarray(scipy.io.loadmat('/media/aeb85/060EE6D90EE6C0B3/Users/aeb85/Desktop/matlabFitData/gainim.mat')['gainim'], dtype=np.float32)

        self.gain_gpu = cuda.mem_alloc(self.dsize)
        #cuda.memcpy_htod(self.gain_gpu, self.gain)

        # FIXME: eventually make the roi size EVEN, i.e. 16x16 so that we use one warp per candidate molecule
        # FIXME: temporarily setting ROIsize to size of test data generated by fittestjig
        #self.ROIsize = int(16)  #int(self.dshape[0])  #int(18)
        self.calcCRLB = np.int32(1)

        self.maxCandCount = 400
        self.dparsZ = np.zeros(6*self.maxCandCount, dtype=np.float32)
        self.dpars = np.zeros_like(self.dparsZ)
        self.dpars_gpu = cuda.mem_alloc(self.dparsZ.size*self.dparsZ.dtype.itemsize)


        #self.CRLBs = np.zeros((6, self.candCount), dtype=np.float32)
        self.CRLBZ = np.zeros(6*self.maxCandCount, dtype=np.float32)
        self.CRLB = np.zeros_like(self.CRLBZ)
        self.CRLB_gpu = cuda.mem_alloc(self.CRLB.size*self.CRLB.dtype.itemsize)

        self.LLHZ = np.zeros(self.maxCandCount, dtype=np.float32)
        self.LLH = np.zeros_like(self.LLHZ)
        self.LLH_gpu = cuda.mem_alloc(self.LLH.size*self.LLH.dtype.itemsize)


    #@classmethod
    def prepvar(self, varmap, flatmap):
        self.varmap = varmap
        print(np.shape(varmap))
        cuda.memcpy_htod_async(self.gain_gpu, np.ascontiguousarray(flatmap, dtype=np.float32), stream=self.vstreamer2)
        #cuda.memcpy_htod_async(self.gtimesv_gpu, self.varmap/flatmap, stream=self.vstreamer1)
        cuda.memcpy_htod_async(self.filter1_gpu, self.dfilterBig, stream=self.dstreamer1)
        cuda.memcpy_htod_async(self.filter2_gpu, self.dfilterSmall, stream=self.dstreamer1)
        cuda.memcpy_htod_async(self.invvar_gpu, varmap, stream=self.dstreamer1)


        #self.varprep(self.invvar_gpu, self.unif1v_gpu, self.unif2v_gpu, self.filter1_gpu, self.filter2_gpu,
        #             self.halfFiltBig, self.halfFiltSmall, self.colsize, block=(int(self.dshape[1]), 1, 1),
        #             grid=(self.rsize, 1), stream=self.vstreamer1)
        self.rfunc_v(self.invvar_gpu, self.unif1v_gpu, self.filter1_gpu,
                   self.halfFiltBig, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer1)

        self.rfunc_v(self.invvar_gpu, self.unif2v_gpu, self.filter2_gpu,
                   self.halfFiltSmall, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer2)

        #self.vstreamer2.synchronize()

        self.cfunc(self.unif1v_gpu, self.filter1_gpu, self.rowsize, self.colsize, self.halfFiltBig,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.vstreamer1)
        self.cfunc(self.unif2v_gpu, self.filter2_gpu, self.rowsize, self.colsize, self.halfFiltSmall,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.vstreamer2)

        self.vstreamer1.synchronize()
        self.vstreamer2.synchronize()

        #cuda.memcpy_dtoh(self.dtarget, self.unif1v_gpu)
        #plt.show(plt.imshow(self.dtarget, interpolation='nearest'))

    def smoothFrame(self, photondat):

        # fixme: do not need to store self.data, just using it for troubleshooting
        self.data = np.ascontiguousarray(photondat, dtype=np.float32)
        cuda.memcpy_htod_async(self.data_gpu, photondat, stream=self.dstreamer1)

        ########################################################################################

        # (float *data, float *invvar, int colsize, int halfFilt)
        #self.dprep(self.data_gpu, self.invvar_gpu, self.colsize, block=(self.csize, 1, 1),
        #           grid=(self.rsize, 1), stream=self.dstreamer1)

        #This should be unnecessary, because calls before and after are in the same stream
        #self.dstreamer1.synchronize()

        ############################# row convolutions ###################################

        #(float *data, float *rowConvBig, float *rowConvSmall, float *filterBig, float *filterSmall, int halfFiltBig, int halfFiltSmall, int colsize)
        """
        self.rfunc(self.data_gpu, self.unif1_gpu, self.unif2_gpu, self.filter1_gpu, self.filter2_gpu,
                   self.halfFiltBig, self.halfFiltSmall, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer1)
                   """
        self.rfunc(self.data_gpu, self.invvar_gpu, self.unif1_gpu, self.gain_gpu, self.filter1_gpu,
                   self.halfFiltBig, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer1)

        self.rfunc(self.data_gpu, self.invvar_gpu, self.unif2_gpu, self.gain_gpu, self.filter2_gpu,
                   self.halfFiltSmall, self.colsize, block=(self.csize, 1, 1),
                   grid=(self.rsize, 1), stream=self.dstreamer2)

        # Wait until stream has finished the row convolutions
        #self.dstreamer1.synchronize()


        ############################# column convolutions ###################################

        # col-convolve the row-convolved variance/gain-weighted data with uniform filters
        self.cfunc(self.unif1_gpu, self.filter1_gpu, self.rowsize, self.colsize, self.halfFiltBig,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.dstreamer1)
        self.cfunc(self.unif2_gpu, self.filter2_gpu, self.rowsize, self.colsize, self.halfFiltSmall,
                   block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.dstreamer2)

        #dstreamer1 does not need to be synced because next call is in that stream.
        #self.dstreamer1.synchronize()
        self.dstreamer2.synchronize()

        ############################# generate and subtract smooth iamges ###################################
        # (float *unifsmalldat,  float *unifsmallvar, float *uniflargedat, float *uniflargevar,int colsize, int halfFilt)
        self.smoothIm(self.unif1_gpu, self.unif1v_gpu, self.unif2_gpu, self.unif2v_gpu, self.colsize,
                      self.halfFiltBig, block=(self.csize, 1, 1), grid=(self.rsize, 1), stream=self.dstreamer1)

        # This stream.sync should be unnecessary because memcpy's are synchronous by default.
        # AND the next call, maxfrow in getCand is also in dstreamer1
        #self.dstreamer1.synchronize()

        #cuda.memcpy_dtoh(self.dtarget, self.unif1_gpu)
        #import matplotlib.pyplot as plt
        #plt.show(plt.imshow(self.dtarget, interpolation='nearest'))


    def getCand(self, thresh=4, ROISize=16):

        # maxFilter size will be 2*halfMaxFilt + 1
        self.halfMaxFilt = np.int32(np.floor(0.5*ROISize) - 1)
        #self.halfMaxFilt = np.int32(7) # fixme this is just to benchmarking vs matlab

        self.maxfrow(self.unif1_gpu, self.maxfData_gpu, self.colsize, self.halfMaxFilt, block=(self.csize, 1, 1),
                     grid=(self.rsize, 1), stream=self.dstreamer1)

        #This should be unnecessary, because calls before and after are in the same stream
        #self.dstreamer1.synchronize()

        self.maxfcol(self.maxfData_gpu, self.colsize, self.halfMaxFilt, block=(self.rsize, 1, 1),
                     grid=(self.csize, 1), stream=self.dstreamer1)

        #This should be unnecessary, because calsl before and after are in the same stream
        #self.dstreamer1.synchronize()
        cuda.memcpy_htod_async(self.candCount_gpu, self.candCountZ, stream=self.dstreamer1)  #rezero the candidate count

        #cuda.memcpy_dtoh(self.dtarget2, self.maxfData_gpu)
        #print(type(self.dtarget[0, 0]))
        #plt.show(plt.imshow((self.dtarget2), interpolation='nearest'))
        #mlMaxFilteredData = np.ascontiguousarray(scipy.io.loadmat('/home/aeb85/PycharmProjects/CandidateDetection/TiffsForMatlabPyCUDAcomparison/quadrant1/peakFinding/maxfilteredData_line61.mat')['maxfilteredData'][:,:,299], dtype=np.int16)
        #mlMaxFilteredData = mlMaxFilteredData.astype(np.float32)
        #plt.show(plt.imshow(mlMaxFilteredData.astype(np.float32) - self.dtarget, interpolation='nearest'))
        #import scipy.ndimage.filters
        #maxfedData = scipy.ndimage.filters.maximum_filter(self.dtarget2, (15, 15))
        #plt.show(plt.imshow(self.dtarget2 >= 0.9999999*maxfedData, interpolation='nearest'))


        #findPeaks(float *unif, float *maxfData, const int thresh, const int colsize)
        self.findpeaks(self.unif1_gpu, self.maxfData_gpu, np.float32(thresh), self.colsize, self.candCount_gpu,
                       self.candPos_gpu, np.int32(0.5*ROISize), block=(self.rsize, 1, 1), grid=(self.csize, 1), stream=self.dstreamer1)

        #self.dstreamer1.synchronize()
        cuda.memcpy_dtoh_async(self.candCount, self.candCount_gpu, stream=self.dstreamer1)
        #print self.candCount

        #  print(self.candCount)


        #cuda.memcpy_dtoh(self.candPos, self.candPos_gpu)
        #print(self.candPos)




    def fitItSlow(self, ROISize=16):
        """
        This function runs David's fast GPU fit, but it reallocates memory for the fit-parameter arrays
        on each run.
        """

        # for astig, Num_Vars = 6, so dpars needs to be at least 6 x candCount
        #self.dpars = np.zeros((6, self.candCount), dtype=np.float32)


        # Re-zero fit outputs
        cuda.memcpy_htod_async(self.dpars_gpu, self.dparsZ, stream=self.dstreamer1)
        cuda.memcpy_htod_async(self.CRLB_gpu, self.CRLBZ, stream=self.dstreamer1)
        cuda.memcpy_htod_async(self.LLH_gpu, self.LLHZ, stream=self.dstreamer1)


        # CRLBs needs to be 6 x candCount, and LogLikelihood needs to be 1xcandCount long
        #self.testROI = np.zeros((ROISize, ROISize), dtype=np.float32)
        #self.testROI_gpu = cuda.mem_alloc(self.testROI.size*self.testROI.dtype.itemsize)
        #cuda.memcpy_htod(self.testROI_gpu, self.testROI)
        #print(np.shape(self.CRLB))
        #print(self.candCount)
        indy = 0
        while indy < self.candCount:
            numBlock = int(np.min([32, self.candCount - indy]))
            #print numBlock
            #self.gaussAstig(self.data_gpu, np.float32(1.4), np.int32(self.ROIsize), np.int32(200),# FIXME: note, second ROIsize would normally be FOV size
            self.gaussAstig(self.data_gpu, np.float32(1.4), np.int32(ROISize), np.int32(200),
                        self.dpars_gpu, self.CRLB_gpu, self.LLH_gpu, self.candCount_gpu, self.invvar_gpu, self.gain_gpu,
                        self.calcCRLB, self.candPos_gpu, np.int32(self.rsize), np.int32(indy),# self.testROI_gpu,
                        block=(ROISize, ROISize, 1), grid=(numBlock, 1), stream=self.dstreamer1)
            indy += numBlock


        cuda.memcpy_dtoh_async(self.dpars, self.dpars_gpu, stream=self.dstreamer1)
        #self.dpars = np.reshape(self.dpars, (6, self.candCount))
        self.dpars = np.reshape(self.dpars, (self.maxCandCount, 6))
        ##self.fitpars = np.reshape(self.dpars, (self.candCount, 6))

        cuda.memcpy_dtoh_async(self.CRLB, self.CRLB_gpu, stream=self.dstreamer1)
        self.CRLB = np.reshape(self.CRLB, (self.maxCandCount, 6))
        cuda.memcpy_dtoh_async(self.LLH, self.LLH_gpu, stream=self.dstreamer1)

        '''
        cuda.memcpy_dtoh(self.testROI, self.testROI_gpu)
        import matplotlib.pyplot as plt
        plt.imshow(self.testROI, interpolation='nearest')
        #plt.scatter()
        #plt.show(plt.imshow(self.data, interpolation='nearest'))
        #plt.scatter(self.dpars[0, 1], self.dpars[0, 0])
        plt.show()
        '''

        return

    def offIt(self):
        """
        This function should be called to clean a detector object context off of the GPU
        """
        self.context.pop()
        return

def normUnifFilter(siz):
    return np.ascontiguousarray((1./siz)*np.ones(siz, dtype=np.float32), dtype=np.float32)