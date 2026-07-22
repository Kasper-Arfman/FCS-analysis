"""
Compute autocorrelation functions (ACFs) from FCS data files and save results to CSV.
This process may be slow, consider running overnight or using a larger bin time for testing.
"""
import numpy as np
from pyjacket import filetools
from tools.read import read_FCS
import pandas as pd
from _script_params import *

OVERWRITE = False

def rstrip_all(s: str, exts: list[str]) -> str:
    """Strip all given extensions from the end of a string."""
    for ext in exts:
        s = s.rstrip(ext)
    return s

EXT = ['.pt3', '.ptu']
fm = filetools.FileManager(DATA_ROOT, ANALYSIS_CACHE)
CONFIG = filetools.read_yaml(CONFIG_PATH)

for date_dir, measurements in CONFIG['measurements'].items():
    print(f"\n\n~~~~~ {date_dir} ~~~~~")
    for measurement in measurements:
        fm.rel_path = f"{date_dir}\\{measurement}"
        if not OVERWRITE and fm.exists('acf.csv', dst=True):
            continue

        print(f"\n== {measurement} ==")
        df = {}
        last_tau = None
        for file_name in fm.iter_dir(ext=EXT):
            print(f" > Processing {file_name}")
            base_name = rstrip_all(file_name, EXT)

            # - Compute ACF
            fcs = read_FCS(fm.src_path(file_name))
            t, counts = fcs.bin_trace(BIN_TIME_ACF)
            tau, acf = fcs.compute_ACF(counts, BIN_TIME_ACF, m=16)

            # - Verify that lag times are consistent
            if last_tau is not None:
                assert np.allclose(tau, last_tau), "Lag times do not match!"
            last_tau = tau

            df[base_name] = acf

        df = pd.DataFrame(df, index=tau)
        fm.write_csv("acf.csv", df)

print('\nFinished Successfully.')