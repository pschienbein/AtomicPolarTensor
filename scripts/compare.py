#!/usr/bin/env python3

import os
import sys
from datetime import datetime
import argparse
import subprocess

import numpy as np

from aptnn.io.xyz import Trajectory
from aptnn.box import Box

import torch

# Home directory of all aptnn related scripts
scriptpath = os.path.dirname(__file__)


# Command line argument parser
parser = argparse.ArgumentParser(description='Script takes configurations providing reference APTs, does the prediction, and compares the prediction with the reference data')
parser.add_argument('--infile', type=str, required=True, help='The input file')
parser.add_argument('--aptnn', type=str, default='aptnn.torch', help='File name of the trained committee nnp (default: aptnn.torch)')
parser.add_argument('--aptout', type=str, default='aptout.xyz', help='Output file name for the predicted APT (default: aptout.xyz)')
parser.add_argument('--scatterout', type=str, default='scatter.dat', help='Output file name, comparing the DFT and NN APTs componentwise (default: scatter.dat)')
parser.add_argument('--rmseout', type=str, default='rmse.dat', help='Output file name containing the total RMSEs (default: rmse.dat)')
parser.add_argument('--varout', type=str, default='uncertainty.dat', help='Output file name containing the score of the prediction, only considering those atoms, where reference data is available (default: uncertainty.dat)')
parser.add_argument('--box', nargs='+', type=float, help='Lattice vectors as "ax ay az bx by bz cx cy cz"', required=True)
parser.add_argument('--num_active_processes', type=int, default=0, help='Number of processes which should be active in parallel; can be used to reduce the amount of memory required, by processing only a subset of committee members in parallel; increases computing time, though (Default: 0, full parallelization)')

args = parser.parse_args()

# 1st step, predict all APTs, contained in the given input file
# this task is outsourced to another script

if not os.path.exists(args.aptout) or not os.path.exists('varout.dat'):
    # how many committee members?
    loaded_data = torch.load(args.aptnn)
    committee_size = len(loaded_data)

    sBox = ' '.join(str(x) for x in args.box)
    cmd = ('mpirun -np %d %s/predict.py '%(committee_size, scriptpath) + 
           '--trajectory %s '%args.infile + 
           '--aptnn %s '%args.aptnn + 
           '--aptout %s '%args.aptout + 
           '--varout varout.dat ' +
           '--num_active_processes %d '%args.num_active_processes + 
           '--box %s '%sBox)
    proc = subprocess.run(cmd.split(), capture_output=True, cwd=os.getcwd())
    fout = open('predict-stdout.log', 'wb')
    fout.write(proc.stdout)
    fout = open('predict-stderr.log', 'wb')
    fout.write(proc.stderr)
    fout.close()

    if proc.returncode != 0:
        print('predict.py failed, check predict-stdout.log and predict-stderr.log', file=sys.stderr)
        exit(proc.returncode)

else:
    print('found', args.aptout, ', skipping prediction of the APT', file=sys.stderr)


# 2nd step
# now that the predicted apt is available, open the reference file and the prediction file and 
# compare all available APTs component wise

#fout_scatter = open(args.scatterout, 'w')
fout_scatter = dict()
fn_scatter = os.path.splitext(args.scatterout)
#print(fn_scatter[0] + '-H' + fn_scatter[1])
#exit()
fout_rmse = open(args.rmseout, 'w')
fout_var = open(args.varout, 'w')


ref_parser = Trajectory(args.infile)
pred_parser = Trajectory(args.aptout)
pred_std_parser = Trajectory('varout.dat')

ref_apt = dict()
mses = dict()
stds = dict()

while True:
    bRef = ref_parser.readNextFrame()
    bPred = pred_parser.readNextFrame()
    bVar = pred_std_parser.readNextFrame()

    if bRef and bPred and bVar:
        ref_atoms = ref_parser.frame.atoms
        pred_atoms = pred_parser.frame.atoms
        pred_atoms_std = pred_std_parser.frame.atoms

        for i in range(len(ref_atoms)):
            if ref_atoms[i].apt is not None:
                symbol = ref_atoms[i].symbol

                if symbol not in mses:
                    mses[symbol] = []
                    fout_scatter[symbol] = open(fn_scatter[0] + '-' + symbol + fn_scatter[1], 'w')

                mses[symbol].append(((ref_atoms[i].apt - pred_atoms[i].apt)**2))

                if symbol not in ref_apt:
                    ref_apt[symbol] = []
                ref_apt[symbol].append(ref_atoms[i].apt)

                for j in range(3):
                    for k in range(3):
                        print(ref_atoms[i].apt[j][k], pred_atoms[i].apt[j][k], file=fout_scatter[symbol])

                
                if symbol not in stds:
                    stds[symbol] = []
                stds[symbol].append(pred_atoms_std[i].apt_std) 
#                stds[symbol]. append(np.mean(pred_atoms_std[i].apt_std)) 

    elif not bRef and not bPred:
        # normal end of the loop 
        break

    else:
        # different number of frames in both files, something went wrong!
        print('ERROR: Number of frames in the prediction and reference file is not equal. Maybe prediction went wrong??', file=sys.stderr)
        exit(1)

total = []
for symbol in mses:
    ref_sd = np.std(ref_apt[symbol], axis=0)

    component_rmse = np.sqrt(np.mean(mses[symbol], axis=0))

#    rel_dev = component_rmse / ref_sd
#    avg_rel_dev = np.mean(rel_dev)

    print('Atom', symbol, file=fout_rmse)
    print('Component SD (DFT)', file=fout_rmse)
    print(ref_sd, file=fout_rmse)
    print('Component RMSE', file=fout_rmse)
    print(component_rmse, file=fout_rmse)
#    print('SCORE (1 - RMSE/SD), component-wise:', 1 - avg_rel_dev, file=fout_rmse)
    print('Overall RMSE:', np.mean(component_rmse), file=fout_rmse)
    print('==========', file=fout_rmse)
    total.extend(mses[symbol])

    tmp = []
    for std in stds[symbol]:
        tmp.append(std / ref_sd)

    meanstd = np.mean(stds[symbol])
    print(symbol, 'mean std:', meanstd, file=fout_var)
    print(symbol, 'mean std (weigted):', np.mean(tmp), file=fout_var)


rmse = np.sqrt(np.mean(total))
print('Total RMSE:', rmse, file=fout_rmse)



