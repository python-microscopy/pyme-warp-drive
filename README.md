# pyme-warp-drive

GPU-accelerated per-pixel SNR & background estimation, candidate molecule detection, and astigmatic Gaussian localization fitting for single-molecule localization microscopy (SMLM) in PYME. 

See [An Integrated Platform for High-Throughput Nanoscopy](https://doi.org/10.1101/606954) for more information.

## Installation

0) Install [PYME](https://python-microscopy.org/doc/Installation/Installation.html)
1) Install CUDA toolkit
2) Install PyCUDA (e.g. 'pip install pycuda')
3) Clone this repository `git clone https://github.com/python-microscopy/pyme-warp-drive`
4) Run 'python setup.py develop' (or install) in the top directory of the repository
5) run 'pytest' in the top directory to check install

## Use

PYME is set up to use warp-drive if it is installed. In PYME, select `AstigGaussGPUFitFR` as the fit type for SMLM localization analysis to use the warp-drive accelerated code. Then, make sure `Calculate percentile background on GPU` is checked in the analysis settings panel.

Developers interested in using warp-drive from 3rd-party code will likely want to focus on the top directory of the repository. We use PyCUDA to compile and create python functions for our CUDA kernels (top level, `.cu` files). The Python code handling memory transfers and the ordering of kernel execution, etc. can be found in the top-level files detector.py (candidate moelcule detection and localization fitting) and buffers.py (pixel-pixel background estimation).

## Licensing
See LICENSE.pdf for licensing information. Alternative licensing may be possible, particularly for per-pixel SNR & background estimation - please contact PYME maintainers at support@python-microscopy.org if that interests you.
