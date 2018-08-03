
from PYME.IO.DataSources.RandomDataSource import DataSource
from PYME.IO.buffers import dataBuffer
from warpDrive.buffers import Buffer
import numpy as np

class IOErrorDataBuffer(object):
    """
    awkwardly holds onto a normal databuffer, but has an attribute used as a flag to trigger IOErrors on
    dataBuffer.getSlice()
    """
    def __init__(self, data_buffer):
        self.data_buffer = data_buffer
        self.dataSource = data_buffer.dataSource
        self.fail = False

    def getSlice(self, ind):
        if self.fail:
            raise IOError('testing GPU buffer handling of IOErrors')

        return self.data_buffer.getSlice(ind)

PERCENTILE = 0.25
IMSIZE_R = 960
IMSIZE_C = 240


ds = DataSource(IMSIZE_R, IMSIZE_C, 100)
# create two buffers for the same datasource, one can fail on command
DBUFF = dataBuffer(ds)
EBUFF = IOErrorDataBuffer(DBUFF)

def get_cpu_background(indices):
    cpu_buffer = np.empty((IMSIZE_R, IMSIZE_C, len(indices)))
    for ind, fi in enumerate(indices):  # g_buf.buffer_length):
        cpu_buffer[:, :, ind] = DBUFF.getSlice(fi)
    cpu_sorted = np.sort(cpu_buffer[:, :, :len(indices)], axis=2)
    index_to_grab = np.int32(max([round(PERCENTILE * len(indices)) - 1, 0]))
    bg_cpu = cpu_sorted[:, :, index_to_grab]
    return bg_cpu


def gpu_cpu_comparison(buffer_length, indices):
    try:
        indices[0]
    except:
        # put indices in a list
        indices = [indices]

    # make GPU buffer
    g_buf = Buffer(DBUFF, percentile=PERCENTILE, buffer_length=buffer_length)

    for bg_indices in indices:
        bg_cpu = get_cpu_background(bg_indices)
        bg_gpu = g_buf.getBackground(bg_indices)

        assert np.array_equal(bg_cpu, bg_gpu)

def simulate_IOError(buffer_length, fail_at):
    """

    Parameters
    ----------
    buffer_length : int
    fail_at : float
        Should be in the range [0, 2]. Specifies the fraction of buffer at which to fail. If less than 1, the data
        buffer will throw an IOError at the gpu buffer before the buffer is full. If larger than 1, the error will be
        thrown after the buffer is full.

    Returns
    -------

    """

    indices = [set(range(bi, bi + buffer_length)) for bi in range(2 * buffer_length)]

    fail_queue = np.zeros_like(indices, dtype=bool)
    # fail before buffer is full, and then after buffer is full, to test both cases
    fail_ind = int(fail_at * buffer_length)
    fail_queue[fail_ind] = True

    # make GPU buffer with fallible data buffer
    g_buf = Buffer(EBUFF, percentile=PERCENTILE, buffer_length=buffer_length)

    ioerror_count = 0
    for bg_indices, fail in zip(indices, fail_queue):
        g_buf.data_buffer.fail = fail  # getSlice will fail if True

        # note that get_cpu_background grabs data from a perfectly functioning data buffer
        bg_cpu = get_cpu_background(bg_indices)
        # our poor gpu buffer, however, does not
        try:
            bg_gpu = g_buf.getBackground(bg_indices)
            # if there is no IOError, then the two calculations should be identical
            assert np.array_equal(bg_cpu, bg_gpu)
        except IOError:
            ioerror_count += 1

    assert ioerror_count < 2

# ----------------- basic tests --------------------- #

def test_32():
    gpu_cpu_comparison(32, set(range(32)))

def test_dynamic():
    gpu_cpu_comparison(41, set(range(41)))

    gpu_cpu_comparison(10, set(range(10)))

def test_underfilled():
    gpu_cpu_comparison(32, set(range(10)))

def test_underfilled_dynamic():
    gpu_cpu_comparison(29, set(range(10)))


def test_series_start():
    buffer_length = 32
    indices = []
    for bi in range(1, buffer_length + 1):
        indices.append(set(range(bi)))

    gpu_cpu_comparison(buffer_length, indices)

def test_recycling():
    buffer_length = 32
    indices = [set(range(buffer_length))]
    for bi in range(1, buffer_length):
        indices.append(set(range(buffer_length, buffer_length + bi)))

    gpu_cpu_comparison(buffer_length, indices)

def test_recycling_with_overlap():
    buffer_length = 32
    indices = [set(range(buffer_length))]
    for bi in range(buffer_length / 2, buffer_length):
        indices.append(set(range(bi)))

    gpu_cpu_comparison(buffer_length, indices)

# ----------------- trickier cases --------------------- #

def test_IOError_on_get_frame():
    simulate_IOError(32, 0.5)
    simulate_IOError(32, 1.5)

# def test_full_series():
#     indices = []
#     for fi in range(32, DBUFF.dataSource.length):
#         start = fi - 32
#         indices.append(set(range(start, fi)))
#
#     gpu_cpu_comparison(32, indices)

