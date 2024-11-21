"""Converts PTU files to ACF data.
Written by Kasper Arfman

Disclaimer:
ACF has not yet been validated. Please confirm for yourself whether 
the computed ACF curve is identical to the one obtained using picoquant software.

NOTE:
 - This script only works for .ptu files. To read pt3 files, please see the read_pt3 script

"""


import ptufile
import numpy as np
import matplotlib.pyplot as plt
import multipletau
import pandas as pd

ROOT_DIR, FILE_NAME = r"D:\Data\FCS\20241120\20241120", r"01_R110_3_7_1_1.pt3"
FILE_PATH = f"{ROOT_DIR}\\{FILE_NAME}"
BIN_TIME = 3e-7  # seconds (ACF conversion can take a while, set this larger for testing)

# == Read file
ptu_data = ptufile.PtuFile(FILE_PATH)
records = ptu_data.decode_records(ptu_data.read_records())
time = records['time'] * ptu_data.global_resolution

# == Bin the time trace data
total_time = time[-1] - time[0]
n_bins = int(total_time // BIN_TIME)
bin_time = total_time / n_bins
counts, bin_edges = np.histogram(time, bins=n_bins)
counts = counts / BIN_TIME  # Units Hz

# == Autocorrelate on a log-scale
tau, acf = multipletau.autocorrelate(counts, m=16, normalize=True)[1:].T
tau *= bin_time

# == Write output to file
df = pd.DataFrame({
    'tau': tau,
    'acf': acf,
})
df.to_csv(f'{FILE_NAME}.csv', index=False)

# == Visualize ACF
if True:
    plt.plot(tau, acf, '.k')
    plt.xscale('log')
    plt.show()