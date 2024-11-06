#!/usr/bin/env python3

# Written by Kit Joll, 31/10/2024

import socket
import argparse
import numpy as np
from mpi4py import MPI
import sys
import warnings
from time import time
import configparser
from ase.io import read
from aptnn.atom import Atom, Frame
from aptnn.box import Box
from aptnn.committee import CommitteeAPTNN
from aptnn.io.xyz import Trajectory, write_conf

# Prequesites for the driver:
# socket
# numpy
# argparse
# mpi4py
# sys
# warnings
# time
# configparser
# ase
# aptnn


warnings.simplefilter("ignore")

description = """
Minimal example of a Python driver connecting to i-PI and exchanging energy, forces, etc.
"""

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def recv_data(sock, data):
    """Fetches binary data from i-PI socket."""
    blen = data.itemsize * data.size
    buf = np.zeros(blen, np.byte)

    bpos = 0
    while bpos < blen:
        timeout = False
        try:
            bpart = 1
            bpart = sock.recv_into(buf[bpos:], blen - bpos)
        except socket.timeout:
            print(" @SOCKET:   Timeout in status recvall, trying again!")
            timeout = True
            pass
        if not timeout and bpart == 0:
            raise RuntimeError("Socket disconnected!")
        bpos += bpart
    if np.isscalar(data):
        return np.frombuffer(buf[0:blen], data.dtype)[0]
    else:
        return np.frombuffer(buf[0:blen], data.dtype).reshape(data.shape)

def send_data(sock, data):
    """Sends binary data to i-PI socket."""

    if np.isscalar(data):
        data = np.array([data], data.dtype)
    buf = data.tobytes()
    sock.send(buf)

HDRLEN = 12  # number of characters of the default message strings

def Message(mystr):
    """Returns a header of standard length HDRLEN."""

    # convert to bytestream since we'll be sending this over a socket
    return str.ljust(str.upper(mystr), HDRLEN).encode()
    
def parse_input_file(input_file):
    """Parse the input file for the driver."""

    config = configparser.ConfigParser()
    config.read(input_file)
    aptnn_config = config['aptnn']  # Separate variable to avoid reassigning config
    atoms = read(aptnn_config['template'])  # Reading template file with ASE
    atomic_string=[atom.symbol for atom in atoms]
    electric_field_vector = [float(i) for i in aptnn_config['electric_field_vector'].split()]  # Convert to list of floats
    model_path = aptnn_config['model_path']
    
    try:
        apt_output = aptnn_config['apt_output']
    except KeyError:
        apt_output = None
        
    try:  
        efield_output = aptnn_config['efield_force_output']
    except KeyError:
        efield_output = None
        
    return model_path, atomic_string, electric_field_vector, apt_output, efield_output
    


def initialize_model(model_file):
    """Load the CommitteeAPTNN model."""
    global net
    try:
        net = CommitteeAPTNN(committee_size=None, model_parameters=None)
        net.load(model_file)
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")
        exit(1)
    print(f"Rank {rank} finished loading model",file=sys.stderr,flush=True)

def run_driver(
    unix=False,
    address="",
    port=12345,
    f_verbose=False,
    sockets_prefix="/tmp/ipi_",
    input_file=""
):
    """Minimal socket client for i-PI."""

    # Opens a socket to i-PI
    if rank==0:
        print('Rank 0 connecting to i-PI',flush=True)
        print('Running using public_APT/scripts/i-pi-driver.py',flush=True)
        if unix:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(sockets_prefix + address)
            except Exception as e:
                print(f"Error connecting to socket: {e}")
                exit(1)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # this reduces latency for the small messages passed by the i-PI protocol
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((address, port))
    else:
        sock=None


    f_init = False
    f_data = False

    # initializes structure arrays
    cell = np.zeros((3, 3), float)
    icell = np.zeros((3, 3), float)
    pos = np.zeros(0, float)

    # initializes return arrays
    pot = 0.0
    force = np.zeros(0, float)
    vir = np.zeros((3, 3), float)
    nat=None

    # Initialize the parameters for all ranks
    model_path = None
    atom_string = None
    electric_field_vec = None
    
    while True:  # ah the infinite loop!

        if rank ==0:
            header = sock.recv(HDRLEN)
            if f_verbose:
                print("Received ", header)
        else:
            header=None
        
        header = comm.bcast(header, root=0)
        
        if header == Message("STATUS"):
            if rank==0:
                # responds to a status request
                if not f_init:
                    sock.sendall(Message("NEEDINIT"))
                elif f_data:
                    sock.sendall(Message("HAVEDATA"))
                else:
                    sock.sendall(Message("READY"))

        elif header == Message("INIT"):
            #try to load the model here
            #read the parameters to run the driver here
            if rank==0:
                itcount=0
                #get driver parameters
                model_path, atom_string, electric_field_vec, apt_output, efield_force_output = parse_input_file(input_file)
                apt_file = open(apt_output, 'w') if apt_output else None
                force_file = open(efield_force_output, 'w') if efield_force_output else None
                print(f'Rank {rank} received model_file: {model_path}', flush=True)
                print(f'Rank {rank} received atom_string: {atom_string}', flush=True)
                print(f'Rank {rank} received electric_field: {electric_field_vec}', flush=True)

            model_path = comm.bcast(model_path, root=0)
            atom_string = comm.bcast(atom_string, root=0)
            electric_field_vec= comm.bcast(electric_field_vec, root=0)
            
            initialize_model(model_path)
            if rank==0:
                # initialization
                rid = recv_data(sock, np.int32())
                initlen = recv_data(sock, np.int32())
                initstr = recv_data(sock, np.chararray(initlen))
                if f_verbose:
                    print(rid, initstr)
                f_init = True  # we are initialized now
            f_init = comm.bcast(f_init, root=0)


        elif header == Message("POSDATA"):
            start_pos_recv_time= time()
            if rank==0:
                # receives structural information
                cell = recv_data(sock, cell)
                icell = recv_data(
                    sock, icell
                )  # inverse of the cell. mostly useless legacy stuff
                nat = recv_data(sock, np.int32())
                if len(pos) == 0:
                    # shapes up the position array
                    pos.resize((nat, 3))
                    force.resize((nat, 3))
                else:
                    if len(pos) != nat:
                        raise RuntimeError("Atom number changed during i-PI run")
                pos = recv_data(sock, pos)
                # here we need to make our config object which will be passed to the driver - note ipi provides the cell and positions in bohr
                # we need to convert them to angstroms
                bohr_to_angstrom = 1/1.88972613289
                pos = pos * bohr_to_angstrom
                cell = cell * bohr_to_angstrom
                
            cell = comm.bcast(cell, root=0)
            pos = comm.bcast(pos, root=0)
            nat = comm.bcast(nat, root=0)

            # create the frame object
            atoms = []
            for i in range(nat):
                atoms.append(Atom(atom_string[i], pos[i]))
            box = Box()
            box.loadFromVectors(cell)
            config = Frame(atoms=atoms, box=box)

            prediction = net.predict([config])
            
            #apply acoustic sum rule correction
            if rank==0:

                pred_apt = prediction['apt']
                pred_var = prediction['std']
                summedtensors = np.sum(pred_apt[0], axis=0)

                for i in range(nat):
                    pred_apt[0][i] -= summedtensors / nat
                    config.atoms[i].apt = pred_apt[0][i]
                    config.atoms[i].apt_std = pred_var[0][i]

                if apt_file:
                    write_conf(apt_file, config.atoms,meta={'i':f'{itcount}'},fmt='pa')

                for i in range(nat):
                    config.atoms[i].frc = np.matmul(np.transpose(pred_apt[0][i]), electric_field_vec)

                if force_file:
                    write_conf(force_file, config.atoms,meta={'i': f'{itcount}'},fmt='pf')

                for i in range(nat):
                    force[i] = config.atoms[i].frc
            
            if rank ==0:
                f_data = True

        elif header == Message("GETFORCE") and rank==0:
            sock.sendall(Message("FORCEREADY"))

            # sanity check in the returned values (catches bugs and inconsistencies in the implementation)
            if not isinstance(force, np.ndarray) and force.dtype == np.float64:
                raise ValueError(
                    "driver returned forces with the wrong type: we need a "
                    "numpy.ndarray containing 64-bit floating points values"
                )

            if not isinstance(vir, np.ndarray) and vir.dtype == np.float64:
                raise ValueError(
                    "driver returned virial with the wrong type: we need a "
                    "numpy.ndarray containing 64-bit floating points values"
                )

            if len(force.flatten()) != len(pos.flatten()):
                raise ValueError(
                    "driver returned forces with the wrong size: number of "
                    "atoms and dimensions must match positions"
                )

            if len(vir.flatten()) != 9:
                raise ValueError(
                    "driver returned a virial tensor which does not have 9 components"
                )

            extras=None
            send_data(sock, np.float64(pot))
            send_data(sock, np.int32(nat))
            send_data(sock, force)
            send_data(sock, vir)
            send_data(sock, np.int32(0))

            f_data = False

            if rank==0:
                end_frc_send_time = time()
                print(f"Driver Iteration {itcount} took: {end_frc_send_time-start_pos_recv_time}", flush=True, file=sys.stderr)
                itcount+=1

        elif header == Message("EXIT"):
            if rank==0:
                print("Received exit message from i-PI. Bye bye!",flush=True, file=sys.stderr)

                if apt_file:
                    apt_file.close()
                if force_file:
                    force_file.close()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-u",
        "--unix",
        action="store_true",
        default=False,
        help="Use a UNIX domain socket.",
    )

    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="localhost",
        help="Host name (for INET sockets) or name of the UNIX domain socket to connect to.",
    )
    
    parser.add_argument(
        "-S",
        "--sockets_prefix",
        type=str,
        default="/tmp/ipi_",
        help="Prefix used for the unix domain sockets. Ignored when using TCP/IP sockets.",
    )
    
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=12345,
        help="TCP/IP port number. Ignored when using UNIX domain sockets.",
    )
    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="",
        help="Input file for the driver, containing parameters for the driver.",
        required=True
    )

    args = parser.parse_args()

    run_driver(
        unix=args.unix,
        address=args.address,
        port=args.port,
        f_verbose=args.verbose,
        sockets_prefix=args.sockets_prefix,
        input_file=args.input
    )

