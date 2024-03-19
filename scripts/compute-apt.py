#!/usr/bin/env python3                                                                                                                                                                                  

import os
import re
import sys
import copy
import argparse

import numpy as np
from aptnn.box import Box
from aptnn.io.xyz import Trajectory, write_frame
from aptnn.atom import Frame


# Home directory of all aptnn related scripts
scriptpath = os.path.dirname(__file__)

# Command line argument parser
parser = argparse.ArgumentParser(description='Script computes APTs given the wannier center output file provided by cp2k')
parser.add_argument('--pos', type=str, required=True, help='Input configurations, NOT displaced!')
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--numderiv', type=str, required=True, help='XYZ file containing the Wannier centers e.g. from cp2k; for each non-displaced input configuration, 6 wannier center configurations are required')
parser.add_argument('--box', nargs='+', type=float, help='Lattice vectors as "ax ay az bx by bz cx cy cz"', required=True)
parser.add_argument('--charges', nargs='+', type=str, help='Charges of each atom kind as (examples): "H:1 O:6 ..."', required=True)
parser.add_argument('--displacement', type=float, default=0.01, help='Displacement for the numerical derivative (default: 0.01 Angstrom)', required=False)
parser.add_argument('--numderiv_s2', type=str, help='wannier centers of S2 spin, in case of a UKS calculation', required=False, default=None)
parser.add_argument('--override-large-apt-warning', type=bool, default=False, required=False, help='Overrides the error when finding an APT with a component larger than 10')
args = parser.parse_args()

###############################################

bPeriodic = True
if len(args.box) != 9:
    print('Requires the periodic lattice in the format "ax ay az bx by bz cx cy cz"', file=sys.stderr)
    exit(1)

print('Input Configurations (NOT displaced!):', args.pos) 
print('Output File:', args.outfile)

box = Box()
lattice = [[args.box[0], args.box[1], args.box[2]], 
           [args.box[3], args.box[4], args.box[5]],
           [args.box[6], args.box[7], args.box[8]]]
print('Box:', lattice)
box.loadFromVectors(lattice)

# charges:
charges = {}
for me in args.charges:
    kv = me.split(':')
    charges[kv[0]] = float(kv[1])
print('Charges:', charges)

# displacement
displacement = args.displacement
print('Displacement:', displacement, 'Angstrom')

if args.numderiv_s2 is not None:
    print('Considering two wannier files, %s and %s'%(args.numderiv, args.numderiv_s2))






########################
# Processing

fout = open(args.outfile, 'w')

config = Trajectory(args.pos)
wannier = Trajectory(args.numderiv)
wannier_S2 = None
if args.numderiv_s2 is not None:
    wannier_S2 = Trajectory(args.numderiv_s2)

while config.readNextFrame():
    # check if the atom id is contained in the meta data:
    if 'atom_id' in config.frame.meta:
        iDisplacedAtom = int(config.frame.meta['atom_id'])
    else:
        # Backward compatibility...
        # extract which atom is displaced
        res = re.search(r'displaced atom (\d+)', config.comment)
        if res is None:
            print('Could not extract displaced atom id from config file', args.pos, file=sys.stderr)
            exit(1)
        iDisplacedAtom = int(res.group(1))
    
    # for each configuration, 6 electronic structure calculations are expected
    frames = []
    for i in range(6):
        if not wannier.readNextFrame():
            print('%s does not contain enough frames; maybe CP2k crashed half way?'%(args.numderiv), file=sys.stderr)
            exit(1)
        frames.append(wannier.atoms)

        if wannier_S2 is not None:
            if not wannier_S2.readNextFrame():
                print('%s does not contain enough frames; maybe CP2k crashed half wat?'%(args.numderiv_s2), file=sys.stderr)
                exit(1)

            for wan_s2 in wannier_S2.atoms:
                if wan_s2.symbol == 'X':
                    frames[i].append(wan_s2)

        if (i - 1) % 2 == 0 and bPeriodic:
            atoms_ref = copy.deepcopy(frames[i])

            # account for pbc 
            for iAtom in range(len(frames[i-1])):

                if frames[i-1][iAtom].symbol == 'X':
                    # this is a bit more involved, since wanniers are not guaranteed to be in the same order 
                    md = 100.0
                    mvd = np.zeros(3)
                    midx = 0
                    for jAtom in range(len(atoms_ref)):
                        if atoms_ref[jAtom].symbol != 'X':
                            continue

                        vd = box.pbc(atoms_ref[jAtom].position - frames[i-1][iAtom].position)                                        
                        d = np.linalg.norm(vd)
                        if d < md:
                            midx = jAtom
                            mvd = vd
                            md = d

                    frames[i][iAtom].position = frames[i-1][iAtom].position + mvd
                    atoms_ref.pop(midx)

                else:
                    # Atoms occur always at the same position in the file -> easy
                    d = box.pbc(frames[i][iAtom].position - frames[i-1][iAtom].position)
                    frames[i][iAtom].position = frames[i-1][iAtom].position + d


                # DBG
                v_dbg = frames[i][iAtom].position - frames[i-1][iAtom].position
                d_dbg = np.linalg.norm(v_dbg)
                if d_dbg > 0.1:
                    print(iAtom)
                    print(frames[i][iAtom], frames[i-1][iAtom])
                    print(frames[i][iAtom].position - frames[i-1][iAtom].position)
                    print(box.pbc(frames[i][iAtom].position - frames[i-1][iAtom].position))

    # compute total dipole moments
    Ms = []
    for frame in frames:
        totcharge = 0
        M = np.zeros(3)
        for atm in frame:
            try:
                M += charges[atm.symbol] * atm.position
                totcharge += charges[atm.symbol]
            except KeyError:
                print('Requested charge of element %s, but not found; correctly initialized? Ensure that all charges relevant are given in the statusfile!'%atm.symbol, file=sys.stderr)
                exit(1)
        Ms.append(M)

        # report non-zero charge, might be an error
        if totcharge != 0:
            print('Total charge is %d. Intended?'%totcharge)                        

#    print('Ms:', Ms) 

    dx = (Ms[0] - Ms[1]) / (2.0 * displacement)
    dy = (Ms[2] - Ms[3]) / (2.0 * displacement)
    dz = (Ms[4] - Ms[5]) / (2.0 * displacement)

#    print(displacement)
#    print(Ms[0])
#    print(Ms[1])
#    print(dx)
#    print(Ms[2])
#    print(Ms[3])
#    print(dy)
#    print(Ms[4])
#    print(Ms[5])
#    print(dz)

    if np.any(np.abs(dx) > 10) or np.any(np.abs(dy) > 10) or np.any(np.abs(dz) > 10):
        if args.override_large_apt_warning:
            print('WARNING: One component of an APT is larger than 10, but continuing anyways!', file=sys.stderr)
        else:
            print('ERROR: One component of an APT is larger than 10, if intended, skip using --override-large-apt-warning flag', file=sys.stderr)
        print('Atom-ID:', iDisplacedAtom, file=sys.stderr)
        print('APT:', dx, dy, dz, file=sys.stderr)

        if not args.override_large_apt_warning:
            exit(1)

    # assign apt to atom
    # dx is the first column of the matrix; but numpy stores matrices in ROW-major order, this must be respected here
    config.frame.atoms[iDisplacedAtom].apt = np.array([ dx[0], dy[0], dz[0], dx[1], dy[1], dz[1], dx[2], dy[2], dz[2] ])
#    conf = config.atoms
#    conf[iDisplacedAtom].apt = np.array([ dx[0], dy[0], dz[0], dx[1], dy[1], dz[1], dx[2], dy[2], dz[2] ])


    # write to the final result file
    write_frame(fout, config.frame)
#    frame = Frame(atoms=conf, box=box)
#    write_frame(fout, frame, config.comment)

#    write_conf(fout, conf, config.comment)

    print('Processed atom', iDisplacedAtom)






