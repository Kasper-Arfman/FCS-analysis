from phconvert import pqreader as pq
import numpy as np
import multipletau
import os
import ptufile

class FCSreader:
    file_path: str
    time: np.ndarray

    def bin_trace(self, bin_time, hertz=True):
        counts, edges = np.histogram(self.time, 
            bins=int((self.time[-1] - self.time[0]) // bin_time)
            )
        time = (edges[:-1] + edges[1:]) / 2
        if hertz: # Counts => Hz
            counts = counts / (time[1] - time[0])  
        return time, counts

    def compute_ACF(self, binned_trace: np.ndarray, bin_time, m=16):
        tau, acf = multipletau.autocorrelate(binned_trace, m=m, normalize=True)[1:].T
        acf += 1
        tau *= bin_time
        return tau, acf


class FCSreaderPT3(FCSreader):

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.time = self.get_time()
    
    def get_time(self):
        time, _, _, meta = pq.load_pt3(self.file_path)
        return time * meta['timestamps_unit']
    

class FCSreaderPTU(FCSreader):
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.time = self.get_time()
    
    def get_time(self):
        ptu_data = ptufile.PtuFile(self.file_path)
        records = ptu_data.decode_records(ptu_data.read_records())
        return records['time'] * ptu_data.global_resolution
    
def read_FCS(file_path: str) -> FCSreaderPT3:
    _, ext = os.path.splitext(file_path)
    reader = {
        '.pt3': FCSreaderPT3,
        '.ptu': FCSreaderPTU,
    }.get(ext, None)

    if reader is None:
        raise NotImplementedError(f'Cannot read file of type {ext}')

    return reader(file_path)