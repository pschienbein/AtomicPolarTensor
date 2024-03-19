
import os
import sys
import copy
from datetime import datetime
import argparse

import numpy as np

from aptnn.io.xyz import Trajectory, write_frame, write_conf

parser = argparse.ArgumentParser(description='Takes a trajectory XYZ file and displaces the respective atom identified with "atom_id" in the comment line')
parser.add_argument('--infile', type=str, help='<Required> XYZ file containing the configurations', required=True)
parser.add_argument('--outfile', type=str, help='File where the displaced configurations are printed (default: displaced.xyz)', default='displaced.xyz')
parser.add_argument('--displacement', type=float, help='Displacement of the atoms for the numerical derivative in angstrom (default: 0.01)', default=0.01)

args = parser.parse_args()

d = args.displacement
d_vectors = [
        [ d, 0, 0 ], [ -d, 0, 0 ],
        [ 0, d, 0 ], [ 0, -d, 0 ],
        [ 0, 0, d ], [ 0, 0, -d ]
        ]

with open(args.outfile, 'w') as fout:
    parser = Trajectory(args.infile)
    while parser.readNextFrame():
        atom_id = int(parser.frame.meta['atom_id'])
        for vd in d_vectors:
            frame = copy.deepcopy(parser.frame)
            frame.atoms[atom_id].position += vd
            write_frame(fout, frame)

