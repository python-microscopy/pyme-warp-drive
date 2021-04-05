from PYME.IO.DataSources.RandomDataSource import DataSource
from PYME.IO.buffers import dataBuffer
from warpdrive.buffers import Buffer
import numpy as np

try:
    from PYME.localization.remFitBuf import CameraInfoManager
    from PYME.IO import MetaDataHandler
    mdh = MetaDataHandler.DictMDHandler()
    mdh['Camera.ReadNoise'] = 1.0
    mdh['Camera.NoiseFactor'] = 1.0
    mdh['Camera.ElectronsPerCount'] = 1.0
    mdh['Camera.TrueEMGain'] = 1.0
    mdh['voxelsize.x'] = 0.7
    mdh['voxelsize.y'] = 0.7
    mdh['Analysis.DetectionFilterSize'] = 3
    mdh['Analysis.ROISize'] = 4.5
    mdh['Analysis.GPUPCTBackground'] = False

    camera_info_manager = CameraInfoManager()
except ImportError:
    print("PYME not installed")
    import pytest
    pytest.skip('python-microscopy environment (PYME) not installed')



def gen_image(p=.95, disp=False):
        from PYME.Acquire.Hardware.Simulator import wormlike2
        from scipy import ndimage

        wc = wormlike2.wiglyFibre(.5e3, 1e2, 1)
        x = np.mod(wc.xp, 64)
        y = np.mod(wc.yp, 66)

        print('Density = ', (1-p)/(.07**2), '/um^2')
        mt = (np.random.rand(len(x)) > p)
        m = np.histogram2d(x[mt], y[mt], [np.arange(0, 64), np.arange(0, 64)])[0].astype('f')
        
        im = ndimage.gaussian_filter(500 * m * (np.random.poisson(4, m.shape) + 1), 2)
        im2 = np.random.poisson(im + 10)
    
        return x[mt], y[mt], im2

def test_repeatability_same_frame():
    from PYME.localization.FitFactories import AstigGaussGPUFitFR
    from PYME.IO import MetaDataHandler
    

    x, _, im = gen_image()
    

    fitter = AstigGaussGPUFitFR.FitFactory(np.atleast_3d(im), mdh)
    results = fitter.FindAndFit(1, cameraMaps=camera_info_manager)
    I_results = np.argsort(results['fitResults']['x0'])

    n_simulated = len(x)
    n_detected = len(results)

    print('Detected: %d, Simulated: %d' % (n_detected, n_simulated))

    for ind in range(100):
        fitter = AstigGaussGPUFitFR.FitFactory(np.atleast_3d(im), mdh)
        res = fitter.FindAndFit(1, cameraMaps=camera_info_manager)
        
        # peak finding inherently orders stochastically, sort results before
        # comparing them
        I_res = np.argsort(res['fitResults']['x0'])
        np.testing.assert_array_equal(res['fitResults'][I_res], 
                                      results['fitResults'][I_results])

def test_repeatability_multiframe():
    from PYME.localization.FitFactories import AstigGaussGPUFitFR
    from PYME.IO import MetaDataHandler
    from PYME.localization.remFitBuf import CameraInfoManager

    n_frame_batch = 50
    res = []
    ims = []
    for ind in range(n_frame_batch):
        _, _, im = gen_image()
        ims.append(im)
        _res = AstigGaussGPUFitFR.FitFactory(np.atleast_3d(im), mdh).FindAndFit(1, cameraMaps=camera_info_manager)
        I = np.argsort(_res['fitResults']['x0'])
        res.append(_res['fitResults'][I])

    n_trials = 10
    for tind in range(n_trials):
        order = np.arange(n_frame_batch, dtype=int)
        np.random.shuffle(order)
        for ind in order:
            _res = AstigGaussGPUFitFR.FitFactory(np.atleast_3d(ims[ind]), mdh).FindAndFit(1, cameraMaps=camera_info_manager)
            I = np.argsort(_res['fitResults']['x0'])
            np.testing.assert_array_equal(_res['fitResults'][I], res[ind])
    