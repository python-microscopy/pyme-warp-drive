
from PYME.IO.DataSources.RandomDataSource import DataSource
from warpDrive.buffers import Buffer
import numpy as np

PERCENTILE = 0.25
IMSIZE_R = 960
IMSIZE_C = 240

class dbuff(object):
    pass

ds = DataSource(IMSIZE_R, IMSIZE_C, 100)
DBUFF = dbuff()
DBUFF.dataSource = ds

def get_cpu_background(indices):
    cpu_buffer = np.empty((IMSIZE_R, IMSIZE_C, len(indices)))
    for ind, fi in enumerate(indices):  # g_buf.buffer_length):
        cpu_buffer[:, :, ind] = DBUFF.dataSource.getSlice(fi)
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


# def test_full_series():
#     indices = []
#     for fi in range(32, DBUFF.dataSource.length):
#         start = fi - 32
#         indices.append(set(range(start, fi)))
#
#     gpu_cpu_comparison(32, indices)

