from warpdrive import *
import numpy as np
import scipy.io
import time

def runTest(plotIt=False, subBkgnd=False, compare=False):
    '''Load data, gain, and variance from .mat files'''
    rawdat = np.ascontiguousarray(scipy.io.loadmat('TestData/imqd1_300.mat')['imqd1_300'], dtype=np.float32)
    #rawdat = rawdat.astype(np.float32)
    #varmap = np.ascontiguousarray(scipy.io.loadmat('/home/aeb85/PycharmProjects/candidatedetection/TiffsForMatlabPyCUDAcomparison/quadrant1/LargeKernel_12pix/varmap.mat')['varmap'][:,:,299], dtype=np.float32)
    varmap = np.ascontiguousarray(scipy.io.loadmat('TestData/varmap.mat')['varmap'], dtype=np.float32)
    gainmap = np.ascontiguousarray(scipy.io.loadmat('TestData/gainim.mat')['gainim'], dtype=np.float32)
    # spoof a flatmap
    electronsPerCount = 1. / np.mean(gainmap)
    # note that PYME-style flatmaps are unitless, whereas in Fang's sCMOS calibration, gain is in units of [ADU/e-]
    flatmap = 1./((gainmap / gainmap.mean()))
    # gainmap = 1./(electronsPerCount*flatmap)

    rawdat = rawdat*gainmap

    halfFilt = np.int32(6) #set half of the filter size
    dfilter1 = normUnifFilter(2*halfFilt)
    halfFilt2 = np.int32(round(0.5*halfFilt))
    dfilter2 = normUnifFilter(2*halfFilt2)

    dshape = (200, 200)
    ditemsize = 4  #each pixel is a float32


    '''Generate warpdrive'''
    _warpdrive = detector(dfilter1, dfilter2)

    _warpdrive.allocateMem(np.shape(rawdat), rawdat.dtype.itemsize)
    _warpdrive.prepvar(varmap, flatmap, electronsPerCount)

    stppwr = 0
    print('Number of trials: %i' % 10**stppwr)
    #cuda.start_profiler()


    telapsed = []
    if subBkgnd:
        dummyBkgnd = np.ascontiguousarray(np.mean(rawdat)*np.ones_like(rawdat))
        print('dummy background nonzero: %s' % np.any(dummyBkgnd))
        for ind in range(0, 10**stppwr):
            t0 = time.time()
            rawdat = np.ascontiguousarray(rawdat)
            _warpdrive.smoothFrame(rawdat, dummyBkgnd)
            _warpdrive.getCand(3.7, 16)  # adjusted threshold in order to run the same nubmer of fits as Matlab. roi16 sets maxfilt size = 15
            _warpdrive.fitItToWinIt()  # : need getCand run each loop in order to test fitItSlow because candidate_positions is zerod after fit each run
            t1 = time.time()
            telapsed.append(t1-t0)
    else:
        for ind in range(0, 10**stppwr):
            t0 = time.time()
            rawdat = np.ascontiguousarray(rawdat)
            _warpdrive.smoothFrame(rawdat)
            _warpdrive.getCand(3.7, 16)
            #_warpdrive.getCand(3.7, 16) #adjusted threshold in order to run the same nubmer of fits as Matlab. roi16 sets maxfilt size = 15
            #_warpdrive.fitItSlow(18) #NOTE: need getCand run each loop in order to test fitItSlow because candidate_positions is zerod after fit each run
            _warpdrive.fitItToWinIt()
            t1 = time.time()
            telapsed.append(t1-t0)

    if compare:
        dummyBkgnd = np.ascontiguousarray(np.ones_like(rawdat))
        rawdat = np.ascontiguousarray(rawdat)
        _warpdrive.smoothFrame(rawdat, dummyBkgnd)
        _warpdrive.getCand(3.7, 16)  # adjusted threshold in order to run the same nubmer of fits as Matlab. roi16 sets maxfilt size = 15
        _warpdrive.fitItToWinIt()  # : need getCand run each loop in order to test fitItSlow because candidate_positions is zerod after fit each run
        dparsBk = np.copy(_warpdrive.fit_res)
        crlbBk = np.copy(_warpdrive.CRLB)
        llhBk = np.copy(_warpdrive.LLH)

        _warpdrive.smoothFrame(rawdat)
        _warpdrive.getCand(3.7, 16)  # adjusted threshold in order to run the same nubmer of fits as Matlab. roi16 sets maxfilt size = 15
        _warpdrive.fitItToWinIt()
        dpars = np.copy(_warpdrive.fit_res)
        crlb = np.copy(_warpdrive.CRLB)
        llh = np.copy(_warpdrive.LLH)

        print(np.array_equal(dparsBk, dpars))
        print(np.array_equal(crlbBk, crlb))
        print(np.array_equal(llhBk, llh))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(llh, edgecolor='red', facecolor='none', label='No background subtraction')
        plt.hist(llhBk, edgecolor='blue', facecolor='none', linestyle='dashed', label='Background subtraction in detection and fit')
        plt.legend(loc=2)
        plt.xlabel('LLH')
        plt.ylabel('Counts')
        plt.show()







    #chan1.offIt()
    #chan2.offIt()

    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################

    print('Mean runtime of %.7f, +- %.7f' % (np.mean(telapsed), np.std(telapsed)))



    #print('Mean runtime of %.7f, +- %.7f' % (np.mean(telapsed), np.std(telapsed)))

    print(_warpdrive.fit_res.shape)
    if plotIt:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(rawdat)
        plt.scatter(np.reshape(_warpdrive.fit_res, (_warpdrive.n_max_candidates_per_frame, 6))[:_warpdrive.n_candidates, 0],
                    np.reshape(_warpdrive.fit_res, (_warpdrive.n_max_candidates_per_frame, 6))[:_warpdrive.n_candidates, 1])
        plt.show()

if __name__ == "__main__":
    import argparse

    # handle inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='plotIt', action='store_true',
                        help='Plot results')
    parser.add_argument('-b', dest='subBkgnd', action='store_true')  # note that the background being subtracted is fake
    parser.add_argument('-c', dest='compare', action='store_true')  # compare bkgnd subtracted with non background subtracted
    args = parser.parse_args()
    runTest(args.plotIt, args.subBkgnd, args.compare)

