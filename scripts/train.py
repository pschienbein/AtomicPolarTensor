#!/usr/bin/env python3                                                                                                                                                                                  

import os
import sys
import torch
import numpy as np

import argparse

from aptnn.committee import CommitteeAPTNN
from aptnn.parameters import ModelParameters
from aptnn.io.xyz import Trajectory
from aptnn.box import Box
from aptnn.util import estimate_num_neighbors

#    atom_types: list 
#    irreps_in: str = field(init=False) # irreps_in is generated later...
#    irreps_out: str = field(default="1x0e+1x1e+1x2e", init=False)
#    radial_cutoff: float = 6.0
#    num_neighbors: float = 90.0
#    num_nodes: float = 5.0
#    pool_nodes: bool = False
#    lmax: int = 2
#    num_neurons: int = 50
#    num_layers: int = 3
#    change_of_basis: torch.tensor = field(init=False)

parser = argparse.ArgumentParser(description='Script trains a new (committee) APTNN on given training data')
parser.add_argument('--inputfn', type=str, help='<Required> XYZ file containing the training data', required=True)
parser.add_argument('--box', nargs='+', type=float, help='<Required> Lattice vectors as "ax ay az bx by bz cx cy cz"', required=True)
parser.add_argument('--outfn', type=str, help='Output file name containing the trained APTNN (default: aptnn.torch)', default='aptnn.torch')
parser.add_argument('--committee_size', type=int, default=8, help='Number of members in the committee (Default: 8)')
parser.add_argument('--device_map', type=str, default='', help='Comma separated list of cuda devices assigning each committee rank its node (Default: '', means auto)')
parser.add_argument('--fixed_seed', action='store_const', default=False, const=True, help='Fixes the seed for separating test/training set (toggle, default: off)')
parser.add_argument('--lmax', type=int, default=2, help='NN architecture: Maximal l for spherical harmonics (Default: 2)')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (Default: 100)')
parser.add_argument('--num_features', type=int, default=30, help='NN architecture: Number of features per node (Default: 30)')
parser.add_argument('--num_mp_layers', type=int, default=2, help='NN architecture: Number of message passing layers (Default: 2)')
parser.add_argument('--num_neighbors', type=int, default=0, help='NN architecture: Number of neighbors of each atom. Will be calculated if set to 0 (Default: 0)')
parser.add_argument('--num_active_processes', type=int, default=0, help='Number of processes which should be active in parallel; can be used to reduce the amount of memory required, by processing only a subset of committee members in parallel; increases computing time, though (Default: 0, full parallelization)')
parser.add_argument('--radial_cutoff', type=float, default=6, help='Interatomic cutoff radius for the graph edges (Default: 6 Angstrom)')
args = parser.parse_args()

print('train_committee.py - given command line arguments:')
# print all arguments: 
for arg in vars(args):
    print(arg, getattr(args, arg))

if len(args.box) != 9:
    exit('Requires the periodic lattice in the format "ax ay az bx by bz cx cy cz"')

box = Box()
lattice = [[args.box[0], args.box[1], args.box[2]], 
           [args.box[3], args.box[4], args.box[5]],
           [args.box[6], args.box[7], args.box[8]]]
#print(lattice)
box.loadFromVectors(lattice)

atom_types = set()
data = []
parser = Trajectory(args.inputfn)
while parser.readNextFrame():
    if parser.frame.box is None:
        parser.frame.box = box
#        print("overwritten", end=' ')

    data.append(parser.frame)
    for symb in parser.get_unique_symbols():
        atom_types.add(symb)

atom_types=sorted(atom_types)

param = ModelParameters(atom_types=list(atom_types))
param.radial_cutoff = args.radial_cutoff
param.num_features = args.num_features
param.num_layers = args.num_mp_layers
param.lmax = args.lmax

if args.num_neighbors == 0:
    num_neighbors = estimate_num_neighbors(param, data)
    print('estimated num_neighbors to be', num_neighbors)
    param.num_neighbors = num_neighbors
else:
    param.num_neighbors = args.num_neighbors

# Fix the seed for all random operations, e.g. separating test/training set 
# only if requested
if args.fixed_seed:
    np.random.seed(0)

# device map
device_map = None
if args.device_map != '':
    device_map = args.device_map.split(',')

# Create network and do the training
net = CommitteeAPTNN(committee_size=args.committee_size, model_parameters=param, device_map=device_map)
net.train(data, args.num_epochs, args.num_active_processes)
net.save(args.outfn)

