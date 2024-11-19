import ptufile
import numpy as np
import matplotlib.pyplot as plt
import multipletau
import pandas as pd

FILE_PATH = r'D:\Data\FCS\20240807\20240807.sptw\DBD LS 100nM _29\DBD LS 100nM _29_3_1_1.ptu'
BIN_TIME = 5e-7  # seconds

# == Read file
ptu_data = ptufile.PtuFile(FILE_PATH)
records = ptu_data.decode_records(ptu_data.read_records())
time = records['time'] * ptu_data.global_resolution

# == Bin the time trace data
total_time = time[-1] - time[0]
n_bins = int(total_time // BIN_TIME)
counts, bin_edges = np.histogram(time, bins=n_bins)
counts = counts / BIN_TIME  # Units Hz

# == Autocorrelate on a log-scale
tau, acf = multipletau.autocorrelate(counts, m=16, normalize=True)[1:].T

# == Write output to file
df = pd.DataFrame({
    't': tau,
    'y': acf,
})
df.to_csv('acf.csv', index=False)

# == Visualize ACF
if True:
    plt.plot(tau, acf, '.k')
    plt.xscale('log')
    plt.show()