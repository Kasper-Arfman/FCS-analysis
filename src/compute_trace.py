from pyjacket import filetools
from tools.read import read_FCS
from _script_params import *

def rstrip_all(s: str, exts: list[str]) -> str:
    """Strip all given extensions from the end of a string."""
    for ext in exts:
        s = s.rstrip(ext)
    return s

EXT = ['.pt3', '.ptu']
OVERWRITE = False

YAML = filetools.read_yaml(CONFIG_PATH)
FM = filetools.FileManager(DATA_ROOT, ANALYSIS_CACHE)


for date_dir, measurements in YAML['measurements'].items():
    print(f"\n\n~~~~~ {date_dir} ~~~~~")
    for measurement in measurements:
        print(f"\n== {measurement} ==")
        dir_path = FM.src_path(measurement, folder=date_dir)

        last_t = None
        for file_name in FM.iter_dir(dir_path, EXT):
            FM.rel_path = f"{date_dir}\{measurement}"
            base_name = rstrip_all(file_name, EXT)

            if not OVERWRITE and FM.exists(f"{base_name}.pkl", dst=True):
                continue

            print(f" > Processing {file_name}")
            fcs = read_FCS(FM.src_path(file_name, dir_path))
            t, Hz = fcs.bin_trace(BIN_TIME_PCD, hertz=False)
            FM.write_pickle(f"{base_name}.pkl", (t, Hz))

        FM.rel_path = ''

print('\nFinished Successfully.')