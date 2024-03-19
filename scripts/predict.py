#!/usr/bin/env python3                                                                                                                                                                                  

import os
from datetime import datetime
import argparse
import traceback

import numpy as np
import torch
import sys

from mpi4py import MPI

from aptnn.committee import CommitteeAPTNN
from aptnn.io.xyz import Trajectory, write_frame
from aptnn.box import Box

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

parser = argparse.ArgumentParser(description='Predicts APTs along a given trajectory using a previously trained APTNN')
parser.add_argument('--box', nargs='+', type=float, help='<Required> Lattice vectors as "ax ay az bx by bz cx cy cz"', required=True)
parser.add_argument('--trajectory', type=str, help='<Required> The trajectory for which the APTs are to be predicted', required=True)
parser.add_argument('--aptnn', type=str, default='aptnn.torch', help='File name of the trained (committee) APTNN (Default: aptnn.torch)')
parser.add_argument('--aptout', type=str, default='apt.xyz', help='Output file name for the APT (Default: apt.xyz)')
parser.add_argument('--varout', type=str, default='', help='Output file name for the per frame variance (Default: empty)')
parser.add_argument('--batch_size', type=int, default=8, help='Number of frames to process in one batch (Default: 8)')
parser.add_argument('--begin', type=int, default=0, help='Frame to start with, used for skipping frames in the trajectory file (Default: 0)')
parser.add_argument('--end', type=int, default=0, help='Frame to end with, used to stop predicting before the end of the file is reached (Default: 0, means disabled)')
parser.add_argument('--device_map', type=str, default='', help='Comma separated list of cuda devices assigning each committee rank its node (Default: '', means auto)')
parser.add_argument('--file_prefix', type=str, help='Optional prefix which is added to all output files (Default: empty)', default='')
parser.add_argument('--num_active_processes', type=int, default=0, help='Number of processes which should be active in parallel; can be used to reduce the amount of memory required, by processing only a subset of committee members in parallel; increases computing time, though (Default: 0, full parallelization)')

args = parser.parse_args()

print('predict.py - given command line arguments:')
# print all arguments: 
for arg in vars(args):
    print(arg, getattr(args, arg))

# device map
device_map = None
if args.device_map != '':
    device_map = args.device_map.split(',')

net = CommitteeAPTNN(committee_size=None, model_parameters=None, device_map=device_map)
net.load(args.aptnn)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if len(args.box) != 9:
    exit('Requires the periodic lattice in the format "ax ay az bx by bz cx cy cz"')

box = Box()
lattice = [[args.box[0], args.box[1], args.box[2]], 
           [args.box[3], args.box[4], args.box[5]],
           [args.box[6], args.box[7], args.box[8]]]
print(lattice)
box.loadFromVectors(lattice)

batch_size = args.batch_size

fn = args.trajectory
# secure the whole process
try: 
    if rank == 0:
        aptout = open(args.file_prefix + args.aptout, 'w')
        varout = None
        if args.varout != '':
            varout = open(args.file_prefix + args.varout, 'w')

        start = datetime.now()
        print('predict started:', start)

        iFrame = 0
        parser = Trajectory(fn)
        data = []
        bContinue = True

        # move to the starting position, if requested
        if args.begin > 0:
            print('moving to', args.begin - 1)
            parser.moveTo(args.begin)         

        # loop until EOF is not reached or there is still data in the array
        while bContinue or len(data) > 0:

            # read the next frame
            bContinue = parser.readNextFrame()
            if bContinue:
                parser.frame.box = box
                data.append(parser.frame)

            
            # early stopping? 
            if args.end > 0 and parser.frameno >= args.end:
                bContinue = False
                print('requested early stopping (--end) at', args.end, 'processing last configurations in buffer and wrapping up...', flush=True)

            
            # start processing if the number of frames in the data array matches the requested batch size, or EOF has been reached.
            if (len(data) == batch_size) or (not bContinue and len(data) > 0):
                res = net.predict(data, args.num_active_processes)

                for iConf in range(len(data)):
                    maxitem = {}

                    for iAtom in range(len(data[iConf].atoms)):
                        symbol = data[iConf].atoms[iAtom].symbol

                        data[iConf].atoms[iAtom].apt = res['apt'][iConf][iAtom]
                        data[iConf].atoms[iAtom].apt_std = res['std'][iConf][iAtom]

                    write_frame(aptout, data[iConf], fmt='pa')
                    if varout is not None:
                        write_frame(varout, data[iConf], fmt='ps')

                    iFrame += 1
                data = []

#                aptout.flush()
#                varout.flush()

        # this ends the loop for all other processes
        net.predict(None)

        timediff = datetime.now() - start
        timediff_sec = timediff.total_seconds()
        print('predict end:', datetime.now(), ';', timediff, flush=True)
        print('predict processed', iFrame, 'frames', flush=True)
        print('predict performance:', iFrame / float(timediff_sec), 'Frames/second')



    else:
        while net.predict([]):
            pass

except Exception as e:
    # print message
    print('Exception caught top level in rank', rank, file=sys.stderr)
    print(str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print('--------', file=sys.stderr)
    print('Aborting', file=sys.stderr)

    comm.Abort(1)
    exit(1)


print('Rank', rank, 'exiting', flush=True)




