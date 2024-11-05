import numpy as np

import sys
import re

from aptnn.atom import Atom, Frame
from aptnn.box import Box

# Simplistic reader of xyz data 

class Trajectory:
    def __init__(self, filename):
        self.fin = open(filename, "r")
        self.frameno = 0
        self.atoms = []
        self.comment = ""
        self.filename = filename
        
        self.comment_pattern_compat_check = re.compile( r'box\s*=\s*"([^"]*?)"(?:\s+|$)')
        self.comment_pattern_compat = re.compile(r'(\w+)\s*=\s*"([^"]*?)"(?:\s+|$)')
        self.comment_pattern = re.compile(r'(\w+)\s*=\s*([^,]+)(?:\s*,\s*|$)')

    def moveTo(self, nFrame):
        # very simplistic, might be optimized lateron
        # currently, the file MUST always contain the same number of atoms

        if len(self.atoms) == 0:
            # read any frame
            # Make sure the parser is not at EOF
            if not self.readNextFrame():
                # reset file pointer
                self.fin.seek(0)
                # try again reading next frame
                self.readNextFrame()

        # number of lines to be skipped
        nLinesPerFrame = len(self.atoms) + 2

        # check if starting from zero again or just moving forward
        if nFrame < self.frameno:
            # start reading from 0
            nLines2Skip = nLinesPerFrame * (nFrame - 1)    
            # reset file pointer
            self.fin.seek(0)
        else:
            # no need to start from scratch
            diff = nFrame - self.frameno
            nLines2Skip = nLinesPerFrame * (nFrame - self.frameno - 1)

        # loop over all to be skipped frames
        for i in range(nLines2Skip):
            self.fin.readline()

        # read the relevant frame
        self.frameno = nFrame - 1
        self.readNextFrame()

    def readVector(self, aLine, idx):
        return np.array([float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])])

    def readTensor(self, aLine, idx):
        return np.array([[float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])],
                     [float(aLine[idx+3]), float(aLine[idx+4]), float(aLine[idx+5])],
                     [float(aLine[idx+6]), float(aLine[idx+7]), float(aLine[idx+8])]])

    def readNextFrame(self):
        # reset
        self.atoms = []

        box = None
        lineno=0

        bFmtSet = False
        fmt = []
        meta = {}

        line = self.fin.readline()
        while line:
            line = line.rstrip()

            if lineno == 0:
                natoms = int(line)

            elif lineno == 1:
                self.comment = line

                # backward compatibility:
                # this code can go at some point!!!
                matches = self.comment_pattern_compat_check.findall(line)
                if len(matches) > 0:
                    #print('compat mode')
                    matches = self.comment_pattern_compat.findall(line)
                    meta = dict(matches)
                    if 'box' in meta:
                        meta['box'] = meta['box'].replace(',',':')
                        
                else:
                    #print('new mode')
                    matches = self.comment_pattern.findall(line)
                    meta = dict(matches)

                    # remove all quotation marks
                    for key in meta:
                        meta[key] = meta[key].replace('"', '')
                        meta[key] = meta[key].replace('\'', '')

                # check if important keys are present
                bFmtSet = ('fmt' in meta)
                if bFmtSet:
                    fmt = meta['fmt'].split(':')

                if 'box' in meta: 
                    nos = meta['box'].split(':')
                    nos = np.array(nos, dtype=np.float64)
                    lattice = nos.reshape((3,3))
                    box = Box()
                    box.loadFromVectors(lattice)

            else:
                aLine = line.split()

                # NOTE: If there is no fmt string given, ASSUME the format given the number of fields
                if not bFmtSet:
                    fmt = []
                    fmt.append('p')
                    if len(aLine) == 7:
                        fmt.append('v')
                    if len(aLine) == 13:
                        fmt.append('a')
                    if len(aLine) == 16:
                        fmt.append('v')
                        fmt.append('a')

                
                elem = aLine[0]

                p = None
                v = None
                a = None
                stddev = None
                apt_total_std_norm = None
                apt_total_std_unnorm = None
                idx = 1
                try:
                    for c in fmt:
                        if c == 'p':
                            p = np.array([float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])])
                            idx += 3

                        elif c == 'v':
                            v = np.array([float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])])
                            idx += 3

                        elif c == 'a':
                            a = np.array([[float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])],
                                         [float(aLine[idx+3]), float(aLine[idx+4]), float(aLine[idx+5])],
                                         [float(aLine[idx+6]), float(aLine[idx+7]), float(aLine[idx+8])]])
                            idx += 9

                        elif c == 's':
                            stddev = np.array([[float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])],
                                               [float(aLine[idx+3]), float(aLine[idx+4]), float(aLine[idx+5])],
                                               [float(aLine[idx+6]), float(aLine[idx+7]), float(aLine[idx+8])]])
                            idx += 9

                        elif c == 'n':
                            apt_total_std_norm = float(aLine[idx])
                            idx += 1
                        
                        elif c == 'u':
                            apt_total_std_unnorm = float(aLine[idx])
                            idx += 1

                        elif c == 't':
                            # try to read APT
                            # only difference to 'a' is that it is accepted if APT is missing, this is only relevant for training data
                            try:
                                a = np.array([[float(aLine[idx]), float(aLine[idx+1]), float(aLine[idx+2])],
                                             [float(aLine[idx+3]), float(aLine[idx+4]), float(aLine[idx+5])],
                                             [float(aLine[idx+6]), float(aLine[idx+7]), float(aLine[idx+8])]])
                                idx += 9

                            except IndexError:
                                a = None
                                pass

                        else:
                            print('WARNING: unknown fmt character in xyz file: %s'%(c), file=sys.stderr)

                except IndexError:
                    print('ERROR: Trying to read fmt code', c, 'but index out of range. Line:', file=sys.stderr)
                    print('   ', line, file=sys.stderr)
                    print('    in', self.filename, file=sys.stderr)
                    exit(1)

                self.atoms.append(Atom(symbol=elem, position=p, velocity=v, apt=a, apt_std=stddev, apt_total_std_norm=apt_total_std_norm, apt_total_std_unnorm=apt_total_std_unnorm))


            if lineno == natoms + 1:
                self.frameno = self.frameno + 1
                if box is not None:
                    self.frame = Frame(atoms=self.atoms, frameno=self.frameno, meta=meta, box=box)
                else:
                    self.frame = Frame(atoms=self.atoms, frameno=self.frameno, meta=meta)
                return True

            lineno = lineno + 1            
            line = self.fin.readline()


        # EOF reached
        return False

    def get_unique_symbols(self):
        elem = set()
        for atom in self.atoms:
            elem.add(atom.symbol)
        return elem


# simplistic xyz writer
def write_conf(fout, atoms, fmt='', meta=dict()):
    data = ""
    float_fmt = "%26.10f"  # Fixed-point format for all floats

    # if no explicit output format given, use the default behavior, i.e. printing positions, velocities, and APTs
    if fmt == '':
        Napts = 0
        Nv = 0
        for atom in atoms:
            data += f'{atom.symbol} {float_fmt % atom.position[0]} {float_fmt % atom.position[1]} {float_fmt % atom.position[2]} '
            if atom.velocity is not None:
                Nv += 1
                for elem in atom.velocity:
                    data += f'{float_fmt % elem} '

            if atom.apt is not None:
                Napts += 1
                for elem in atom.apt.flatten():
                    data += f'{float_fmt % elem} '

            data += '\n'

        attributes = ['p']
        if Nv == len(atoms):
            attributes.append('v')
        if Napts == len(atoms):
            attributes.append('a')
        elif Napts > 0:
            attributes.append('t')
        meta['fmt'] = ':'.join(attributes)

    else:
        if fmt.find(':') >= 0:
            fmt = fmt.split(':')
        else:
            fmt = [*fmt]

        for atom in atoms:
            data += f'{atom.symbol} {float_fmt % atom.position[0]} {float_fmt % atom.position[1]} {float_fmt % atom.position[2]} '
            for c in fmt:
                if c == 'v':
                    if atom.velocity is not None:
                        for elem in atom.velocity:
                            data += f'{float_fmt % elem} '
                    else:
                        print('WARNING: Explicitly requested writing velocities, but not present', file=sys.stderr)

                if c == 'a':
                    if atom.apt is not None:
                        for elem in atom.apt.flatten():
                            data += f'{float_fmt % elem} '
                    else:
                        print('WARNING: Explicitly requested writing APTs, but not present', file=sys.stderr)

                if c == 't':
                    if atom.apt is not None:
                        for elem in atom.apt.flatten():
                            data += f'{float_fmt % elem} '

                if c == 's':
                    if atom.apt_std is not None:
                        for elem in atom.apt_std.flatten():
                            data += f'{float_fmt % elem} '
                    else:
                        print('WARNING: Explicitly requested writing APT standard deviations, but not present', file=sys.stderr)

                if c == 'u':
                    if atom.apt_total_std_unnorm is not None:
                        data += f'{float_fmt % atom.apt_total_std_unnorm} '
                    else:
                        print('WARNING: Explicitly requested writing total unnormalized APT standard deviation, but not present', file=sys.stderr)

                if c == 'n':
                    if atom.apt_total_std_norm is not None:
                        data += f'{float_fmt % atom.apt_total_std_norm} '
                    else:
                        print('WARNING: Explicitly requested writing total normalized APT standard deviation, but not present', file=sys.stderr)
                
                if c == 'f':
                    frc_float_fmt = '%26.10e' # scientific notation same precision
                    if atom.frc is not None:
                        for elem in atom.frc:
                            data += f'{frc_float_fmt % elem} '

            data += '\n'

        meta['fmt'] = ':'.join(fmt)

    # Write number of atoms and metadata comment line
    print(len(atoms), file=fout)
    comment = ', '.join([f'{key}={value}' for key, value in meta.items()])
    fout.write(comment + '\n')

    # Write the data to the file
    fout.write(data)


def write_frame(fout, frame, fmt=''):
    meta = frame.meta
    float_fmt = "%26.10f"  # Fixed-point format for box values

    if frame.box is not None:
        lattice = frame.box.getLatticeVectors()
        boxval = ':'.join([float_fmt % val for row in lattice for val in row])
        meta['box'] = boxval

    write_conf(fout, frame.atoms, meta=meta, fmt=fmt)

    pass
