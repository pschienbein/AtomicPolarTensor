
from datetime import datetime
import copy

import numpy as np
import torch
import torch.cuda

from aptnn.aptnn import APTNN

def _count_generator(reader):
    b = reader(1024*1024)
    while b:
        yield b
        b = reader(1024*1024)

def linecount(fn):
    with open(fn, 'rb') as fp:
        c_generator = _count_generator(fp.raw.read)
        count = sum(buffer.count(b'\n') for buffer in c_generator)
        return count 


def estimate_num_neighbors(model_parameters, data):
    tmp_net = APTNN(torch.device('cpu'), model_parameters=model_parameters)
    tmp_data = copy.deepcopy(data)
    N = []
    for frame in tmp_data:
        tmp_net.create_graph_from_frame(frame)
        Natoms = len(frame.atoms)
        Nedges = len(frame.edge_src)
        N.append(float(Nedges) / Natoms)

    return int(round(np.mean(N)))


class DynamicHistogram: 
    def __init__(self): 
        self.data = []
        pass

    def push(self, value):
        self.data.append(value)
        if len(self.data) > 1000:
            print(np.std(self.data), len(self.data), len(self.data)**(-1.0/3.0))
            bin_width = 3.5 * np.std(self.data) * len(self.data)**(-1.0/3.0)
            print(bin_width)
            num_bins = (np.max(self.data) - np.min(self.data)) / bin_width
            print(np.max(self.data), np.min(self.data))
            print(num_bins)

            exit()

        pass

class RunningMeanVar:
    def __init__(self, zero = 0):
        self.count = 0
        self.M2 = copy.deepcopy(zero)
        self.mean = copy.deepcopy(zero)
        pass


    def update(self, value):
        self.count += 1
        d = value - self.mean
        self.mean += d / self.count
        d2 = value - self.mean
        self.M2 += (d * d2)

        pass

    def get(self):
        return self.mean, self.M2 / self.count

