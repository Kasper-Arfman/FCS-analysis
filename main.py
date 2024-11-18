
def read_PQ():
    """Read PicoQuant data

    Lifetime data
    Time trace
    """

def get_acf():
    """Obtain the Autocorrelation Function from a time trace"""

def fit_model():
    """Fits data to a model"""

def guess_model():
    """Initial guess for model params"""

def fit_distribution():
    """Fit data to a distribution of lifetimes, using max-entropy method"""


import ptufile
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\Data\FCS\20240828\slide3.sptw\02-mNG_LS_2\02-mNG_LS_2_1_1_1.ptu'
file_path = r'D:\Data\FCS\20240828\slide2.sptw\01-R110_10p_80MHz_MQ_2\01-R110_10p_80MHz_MQ_2_2_1_1.ptu'


ptu_data = ptufile.PtuFile(file_path)
print(ptu_data)
records = ptu_data.read_records()
q = ptu_data.decode_records(records)

# array with
# time 





time = q['time']
time_resolution = ptu_data.global_resolution  # 1.28e-8 seconds/tick
time_seconds = time * time_resolution
total_time = time_seconds[-1] - time_seconds[0]  # seconds
print(f'{total_time = :.2f} seconds')


bin_time = 1e-3  # 1 millisecond
n_bins = int(total_time // bin_time)
print(f"{n_bins = }")

counts, bin_edges = np.histogram(time_seconds, bins=n_bins)

# plt.plot(counts)
# plt.show()

plt.plot(counts)
plt.show()







# dtime
# channel
# marker
