import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyjacket import filetools
from tools.maximum_entropy import MaxEnt
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
        if kwargs.get('D'):
            continue

        print(f"\n== {measurement} - {kwargs} ==")
        FM.rel_path = f"{date_dir}\\{measurement}"

        # Read data
        acfs = FM.read_csv('acf.csv')
        for cycle_name in kwargs.get('exclude', []):
            print(f' - EXCLUDED {cycle_name}')
            acfs = acfs.drop(columns=[cycle_name])
        tau, acf_avg, acf_sem = bundle_df(acfs)
        acf_avg[acf_avg<0] = 0

        # Slice the ACF
        rng = (3*BIN_TIME_ACF < tau) & (tau < 1e-1)
        tau = tau[rng]
        acf_avg = acf_avg[rng] - 1
        acf_sem = acf_sem[rng]

        # == Fit ACF == #
        # - Initial guess
        max_ent = MaxEnt(tau, acf_avg, acf_sem)
        max_ent.set_fitting_space(TAU, alpha=ALPHA)  # increase for more smoothness
        shape_factor = kwargs.get('shape_factor', DEFAULT_SHAPE_FACTOR)
        acf_guess = max_ent.initial_guess((shape_factor, 0))

        # - ACF fit
        max_ent.fit_distribution(FIT_CYCLES)
        acf_fit = max_ent.y_fit
        if True:
            plt.plot(tau, acf_avg, label=measurement)
            plt.fill_between(tau, acf_avg-acf_sem, acf_avg+acf_sem, alpha=0.3)
            # plt.plot(tau, acf_guess, 'g-.')  # initial guess
            plt.plot(tau, acf_fit, 'k--')
            plt.xscale('log')
            plt.legend()
            plt.show()

        # == Size distribution
        # - Get calibrant info
        FM.rel_path = f"{date_dir}"
        CAL = FM.read_json('calibrant.json', dst=True)
        hydrodynamic_diameter = KB*T/((6 * np.pi * ETA) * (CAL['T1'] * CAL['D'])) * TAU * 2e9
        if True:
            # plt.figure(figsize=(6, 4))
            plt.plot(hydrodynamic_diameter, max_ent.dist, label=f"{measurement}")
            plt.xscale('log')
            plt.xlabel(r'$d$ ($nm$)')
            plt.ylabel(r'Weight')
            plt.legend()
            # fm.savefig('MaxEnt_MpARF2.pdf', close=False)
            plt.show()

print('\nFinished Successfully.')