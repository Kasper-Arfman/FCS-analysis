import matplotlib.pyplot as plt
import pandas as pd
from reading.read import read_FCS
import os

EXPERIMENTS = [
    r'D:\Data\FCS\20241120\01_R110\01_R110_3_2_1_1.pt3',  # R110
    # r'D:\Data\FCS\20240828\slide3.sptw\02-mNG_LS_3\02-mNG_LS_3_1_1_1.ptu'
]
BIN_TIME = 2e-7  # seconds

for file_path in EXPERIMENTS:
    root_dir, file_name = os.path.split(file_path)

    # == Read file
    fcs = read_FCS(file_path)

    # Visualize time trace at reduced resolution
    if True:
        time, counts = fcs.bin_trace(0.06067, hertz=True)
        plt.title('Time Trace')
        plt.plot(time, counts/1000)
        plt.ylabel('Intensity [kHz]')
        plt.xlabel('Time [s]')
        plt.show()

    # == Bin the time trace data
    # set a smaller bin_time if you want to plot the time trace (e.g. 0.06)
    time, counts = fcs.bin_trace(BIN_TIME, hertz=True)

    # == Autocorrelate on a log-scale
    # len(acf) ~= m + ilog2(N / m)*(m // 2) + 1
    tau, acf = fcs.compute_ACF(counts, time[1] - time[0])
    tau *= 1000  # [ms]
    
    # == Write output to file
    df = pd.DataFrame({
        'tau': tau,  # [ms]
        'acf': acf,
    })
    df.to_csv(f'{file_name}.csv', index=False)

    # == Visualize ACF
    if True:
        plt.plot(tau, acf, '.k')
        plt.xscale('log')
        plt.show()

print('\nFinished successfully')