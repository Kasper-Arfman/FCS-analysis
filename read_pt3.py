from phconvert import pqreader as pq

ROOT_DIR, FILE_NAME = r"D:\Data\FCS\20241120\20241120", r"01_R110_3_7_1_1.pt3"
FILE_PATH = f"{ROOT_DIR}\\{FILE_NAME}"
BIN_TIME = 3e-7  # seconds

time, _, _, meta = pq.load_pt3(FILE_PATH)
time = time * meta['timestamps_unit']

import numpy as np
import matplotlib.pyplot as plt
import multipletau
import pandas as pd
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