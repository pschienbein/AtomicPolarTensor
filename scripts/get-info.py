
import os
import argparse

import numpy as np
import torch

parser = argparse.ArgumentParser(description='')
parser.add_argument('--aptnn', type=str, default='aptnn.torch', help='File name of the trained committee nnp (default: aptnn.torch)')
args = parser.parse_args()

# Note: This is just to gather information, no need to involve cuda
dat = torch.load(args.aptnn, map_location=torch.device('cpu'))

print('# Model Parameters')
print(dat[0]['model_parameters'])

i = 0
for nn in dat:
    print('# Network', i)
    print('# Means used for normalization:')
    for symbol in nn['model_parameters'].atom_types:
        print(symbol)
        print(nn['norm_mean'][symbol])
    print('# Stddevs used for normalization:')
    for symbol in nn['model_parameters'].atom_types:
        print(symbol)
        print(nn['norm_std'][symbol])

    i+=1

