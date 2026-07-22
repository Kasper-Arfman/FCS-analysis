import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pyjacket import filetools
from tools.models import triplet_diffusion as model, triplet_decay, diffusion3D
from _script_params import *

CONFIG = filetools.read_yaml(CONFIG_PATH)
FM = filetools.FileManager(ANALYSIS_CACHE, ANALYSIS_ROOT)

def bundle_df(df: pd.DataFrame):
    x = np.array(df.index)
    y_avg = np.array(df.mean(axis=1))
    y_sem = np.array(df.sem(axis=1))
    return x, y_avg, y_sem

for date_dir, measurements in CONFIG['measurements'].items():
    print(f"\n\n~~~~~ {date_dir} ~~~~~")
    for measurement, kwargs in measurements.items():
        kwargs = {} if kwargs is None else kwargs

        if not kwargs.get('D'):
            continue

        print(f"\n== {measurement} - {kwargs} ==")
        FM.rel_path = f"{date_dir}\\{measurement}"

        data = FM.read_csv('acf.csv')
        tau, acf_avg, acf_sem = bundle_df(data)
        acf_avg[acf_avg<0] = 0

        # Slice the ACF
        rng = (2.1*BIN_TIME_ACF < tau) & (tau < 1e-1)
        tau = tau[rng]
        acf_avg = acf_avg[rng]
        acf_sem = acf_sem[rng]

        # Fitting
        p0 = (
            1.1/(acf_avg[0]-1),  # N
            1,       # G_inf
            3e-6,    # Ttrip
            0.1,     # Ftrip
            32e-6,   # T1 
            10,      # a
        )
        bounds = (
            [  0,   0,   0,   0,   0,   5], 
            [100,   2, 0.1,   1, 0.1,  20],
        )
        popt, pcov = curve_fit(model, tau, acf_avg, 
                               sigma=acf_sem, 
                               p0=p0, 
                               bounds=bounds)

        if True:
            plt.plot(tau, acf_avg)
            plt.fill_between(tau, acf_avg-acf_sem, acf_avg+acf_sem, alpha=0.3)
            plt.plot(tau, model(tau, *p0), 'g-.')  # initial guess
            plt.plot(tau, model(tau, *popt), 'k--')
            plt.xscale('log')
            plt.show()

        N, G, Ttrip, Ftrip, T1, a = popt
        calibrant_popt = dict(
            N=N,
            G=G,
            Ttrip=Ttrip,
            Ftrip=Ftrip,
            T1=T1,
            a=a,
            D=float(kwargs.get('D', DEFAULT_D_CAL)),
        )
        FM.rel_path = f"{date_dir}"
        FM.write_json('calibrant.json', calibrant_popt)
        
print('\nFinished Successfully.')