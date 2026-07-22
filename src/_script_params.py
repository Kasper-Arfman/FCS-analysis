from os.path import dirname, join
from scipy.constants import k as KB
import numpy as np

CODE_ROOT = dirname(dirname(__file__))
DATA_ROOT = join(CODE_ROOT, 'data')
ANALYSIS_CACHE = join(CODE_ROOT, 'analysis/__cache__')
ANALYSIS_ROOT = join(CODE_ROOT, 'analysis')
CONFIG_PATH = join(CODE_ROOT, 'config.yaml')

FOCUS_MEASUREMENT = None
FOCUS_BUNDLE = None
FOCUS_DATASET = None

BIN_TIME_PCD = 1e-5  # s
BIN_TIME_ACF = 2e-5  # s

T = 294  # 21 C
ETA = 0.9764e-3  # Pa*s (water at 21 C)
TAU = np.geomspace(
    1e-7,  # 1e-6
    30,  # Seconds 
    600)
FIT_CYCLES = 500
ALPHA = 1000e-7

DEFAULT_TAU_CAL = 100e-6
DEFAULT_D_CAL = 1e-10
DEFAULT_SHAPE_FACTOR = 8

if __name__ == "__main__":
    print('\nYou should not run _globals.py directly')