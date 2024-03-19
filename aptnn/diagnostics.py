
import numpy as np

class Histogram:
    def __init__(self, _min = 0, _max = 0.1, bins = 10000):
        self.min = _min
        self.max = _max
        self.bins = bins
        self.data = []
        self.dx = (self.max - self.min) / self.bins
        self.norm = 0
        self.act_min = self.max
        self.act_max = self.min

        for i in range(bins):
            self.data.append(0.0)

    def add(self, d):
        b = int((d - self.min) / self.dx) 

        if b < 0 or b >= self.bins:
            print('warning: not in histogram')
        else:
            self.data[b] += 1
            self.norm += 1

        if self.act_min > d:
            self.act_min = d
        
        if self.act_max < d:
            self.act_max = d

        pass


    def mean_std(self):
        s = 0
        w = 0
        for i in range(self.bins):
            weight = self.data[i]
            val = self.dx * i + self.min
            s += val * weight 
            w += weight
        
        if w > 0:
            mean = (s / w) 

            s = 0
            nw = 0
            for i in range(self.bins):
                weight = self.data[i]
                val = self.dx * i + self.min
                s += weight * (val - mean)**2
                if weight > 1e-8:
                    nw += 1

            std = np.sqrt(s / w)
        else:
            mean = 0
            std = 0

        return mean, std

    def min_max(self):
        return self.act_min, self.act_max
        

    def write(self, fn):
        with open(fn, 'w') as fout:
            if self.norm == 0:
                self.norm = 1
            for i in range(self.bins):
                x = self.dx * i
                fout.write('%e %e %e\n'%(self.dx * i + self.min, self.data[i], self.data[i] / self.norm))



