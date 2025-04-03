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
# mpi4py
# sys
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

def recv_data(sock, data, f_verbose=False):
    """Fetches binary data from i-PI socket."""
    blen = data.itemsize * data.size
    buf = np.zeros(blen, np.byte)

    bpos = 0
    while bpos < blen:
        timeout = False
        try:
            bpart = sock.recv_into(buf[bpos:], blen - bpos)
            if f_verbose:
                print(f" @SOCKET:   recv_into -> bpart={bpart}, bpos={bpos}, need={blen - bpos}", flush=True)
        except socket.timeout:
            if f_verbose:
                print(" @SOCKET:   Timeout in status recvall, trying again!", flush=True)
            timeout = True
            pass
        if not timeout and bpart == 0:
            raise RuntimeError("Socket disconnected!")
        bpos += bpart
    if np.isscalar(data):
        return np.frombuffer(buf[0:blen], data.dtype)[0]
    else:
        return np.frombuffer(buf[0:blen], data.dtype).reshape(data.shape)

def send_data(sock, data, f_verbose=False):
    """Sends binary data to i-PI socket."""
    if np.isscalar(data):
        data = np.array([data], data.dtype)
    buf = data.tobytes()
    sock.send(buf)
    if f_verbose:
        print(f" @SOCKET:   sent {len(buf)} bytes of data", flush=True)

HDRLEN = 12  # number of characters of the default message strings

def Message(mystr):
    """Returns a header of standard length HDRLEN."""
    return str.ljust(str.upper(mystr), HDRLEN).encode()

def parse_input_file(input_file, f_verbose=False):
    """Parse the input file for the driver."""
    if f_verbose:
        print(f" @DRIVER:   Parsing input file: {input_file}", flush=True)

    config = configparser.ConfigParser()
    config.read(input_file)
    aptnn_config = config['aptnn']  # Separate variable to avoid reassigning config
    atoms = read(aptnn_config['template'])  # Reading template file with ASE
    atomic_string = [atom.symbol for atom in atoms]
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

    if f_verbose:
        print(f" @DRIVER:   model_path={model_path}", flush=True)
        print(f" @DRIVER:   template has {len(atoms)} atoms: {atomic_string}", flush=True)
        print(f" @DRIVER:   electric_field_vector={electric_field_vector}", flush=True)
        print(f" @DRIVER:   apt_output={apt_output}, efield_force_output={efield_output}", flush=True)

    return model_path, atomic_string, electric_field_vector, apt_output, efield_output


def initialize_model(model_file, f_verbose=False):
    """Load the CommitteeAPTNN model."""
    global net
    if f_verbose:
        print(f" @DRIVER:   Initializing model from file: {model_file}", flush=True)
    try:
        net = CommitteeAPTNN(committee_size=None, model_parameters=None)
        net.load(model_file)
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}", flush=True)
        exit(1)
    if f_verbose:
        print(f" @DRIVER:   Model loaded successfully.", flush=True)


def run_driver(
    unix=False,
    address="",
    port=12345,
    f_verbose=False,
    sockets_prefix="/tmp/ipi_",
    input_file=""
):
    """Minimal socket client for i-PI."""

    if rank == 0 and f_verbose:
        print(" @DRIVER:   Starting run_driver on rank 0", flush=True)
        print(f" @DRIVER:   unix={unix}, address={address}, port={port}", flush=True)
        print(f" @DRIVER:   sockets_prefix={sockets_prefix}", flush=True)
        print(f" @DRIVER:   input_file={input_file}", flush=True)
        print(" @DRIVER:   Connecting to i-PI ...", flush=True)

    # Opens a socket to i-PI
    if rank == 0:
        if unix:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(sockets_prefix + address)
            except Exception as e:
                print(f"Error connecting to socket: {e}")
                exit(1)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # this reduces latency for the small messages passed by i-PI protocol
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((address, port))

        if f_verbose:
            print(" @DRIVER:   Successfully connected to i-PI socket.", flush=True)
            print(" @DRIVER:   Running using public_APT/scripts/i-pi-driver.py", flush=True)
    else:
        sock = None

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
    nat = None

    # Initialize the parameters for all ranks
    model_path = None
    atom_string = None
    electric_field_vec = None

    while True:  # infinite loop
        if rank == 0:
            if f_verbose:
                print(" @DRIVER:   Waiting for next header from i-PI...", flush=True)
            header = sock.recv(HDRLEN)
            if f_verbose:
                print(f" @DRIVER:   Received header: {header}", flush=True)
        else:
            header = None

        header = comm.bcast(header, root=0)

        # ---------------------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------------------
        if header == Message("STATUS"):
            if f_verbose and rank ==0:
                print(" @DRIVER:   Got STATUS request", flush=True)
            if rank == 0:
                if not f_init:
                    sock.sendall(Message("NEEDINIT"))
                    if f_verbose:
                        print(" @DRIVER:   Sent NEEDINIT", flush=True)
                elif f_data:
                    sock.sendall(Message("HAVEDATA"))
                    if f_verbose:
                        print(" @DRIVER:   Sent HAVEDATA", flush=True)
                else:
                    sock.sendall(Message("READY"))
                    if f_verbose:
                        print(" @DRIVER:   Sent READY", flush=True)

        # ---------------------------------------------------------------------
        # INIT
        # ---------------------------------------------------------------------
        elif header == Message("INIT"):
            if f_verbose and rank ==0:
                print(" @DRIVER:   Got INIT request", flush=True)
            if rank == 0:
                itcount = 0
                # get driver parameters
                model_path, atom_string, electric_field_vec, apt_output, efield_force_output = parse_input_file(
                    input_file, f_verbose=f_verbose
                )
                apt_file = open(apt_output, 'w') if apt_output else None
                force_file = open(efield_force_output, 'w') if efield_force_output else None

            # broadcast to all ranks
            model_path = comm.bcast(model_path, root=0)
            atom_string = comm.bcast(atom_string, root=0)
            electric_field_vec = comm.bcast(electric_field_vec, root=0)

            initialize_model(model_path, f_verbose=f_verbose)

            if rank == 0:
                # initialization
                if f_verbose:
                    print(" @DRIVER:   Receiving rid and initstr from i-PI", flush=True)
                rid = recv_data(sock, np.int32(), f_verbose=f_verbose)
                initlen = recv_data(sock, np.int32(), f_verbose=f_verbose)
                initstr = recv_data(sock, np.chararray(initlen), f_verbose=f_verbose)
                if f_verbose:
                    print(" @DRIVER:   Received rid=", rid, ", initstr=", initstr, flush=True)

                f_init = True  # we are initialized now
            f_init = comm.bcast(f_init, root=0)
            if f_verbose and rank ==0:
                print(f" @DRIVER:   Initialization complete. f_init={f_init}", flush=True)

        # ---------------------------------------------------------------------
        # POSDATA
        # ---------------------------------------------------------------------
        elif header == Message("POSDATA"):
            if f_verbose and rank ==0:
                print(" @DRIVER:   Got POSDATA request", flush=True)
            start_pos_recv_time = time()

            if rank == 0:
                if f_verbose:
                    print(" @DRIVER:   Receiving cell, icell, nat, pos from i-PI", flush=True)
                cell = recv_data(sock, cell, f_verbose=f_verbose)
                icell = recv_data(sock, icell, f_verbose=f_verbose)  # legacy stuff
                nat = recv_data(sock, np.int32(), f_verbose=f_verbose)

                # (Re)shape pos and force arrays if needed
                if len(pos) == 0:
                    pos.resize((nat, 3))
                    force.resize((nat, 3))
                else:
                    if len(pos) != nat:
                        raise RuntimeError("Atom number changed during i-PI run")

                pos = recv_data(sock, pos, f_verbose=f_verbose)

                if f_verbose:
                    print(f" @DRIVER:   Received positions for {nat} atoms", flush=True)

                # convert from bohr to angstrom
                bohr_to_angstrom = 1 / 1.88972613289
                pos = pos * bohr_to_angstrom
                cell = cell * bohr_to_angstrom

            # broadcast data to all ranks
            cell = comm.bcast(cell, root=0)
            pos = comm.bcast(pos, root=0)
            nat = comm.bcast(nat, root=0)

            if f_verbose and rank != 0:
                print(f" @DRIVER:   [rank {rank}] Received broadcast pos/cell for {nat} atoms", flush=True)

            # create the Frame object
            atoms = []
            for i in range(nat):
                atoms.append(Atom(atom_string[i], pos[i]))
            box = Box()
            box.loadFromVectors(cell)
            config = Frame(atoms=atoms, box=box)

            # net.predict: You may or may not want to do this only on rank=0,
            # but for demonstration, let's call it on all ranks here.
            if f_verbose:
                print(f" @DRIVER:   [rank {rank}] Calling net.predict(...) on frame", flush=True)
            prediction = net.predict([config])

            # apply acoustic sum rule correction
            if rank == 0:
                if f_verbose:
                    print(" @DRIVER:   Applying acoustic sum rule correction", flush=True)
                pred_apt = prediction['apt']
                pred_var = prediction['std']
                summedtensors = np.sum(pred_apt[0], axis=0)

                for i in range(nat):
                    pred_apt[0][i] -= summedtensors / nat
                    config.atoms[i].apt = pred_apt[0][i]
                    config.atoms[i].apt_std = pred_var[0][i]

                if apt_file:
                    write_conf(apt_file, config.atoms, meta={'i': f'{itcount}'}, fmt='pa')

                # compute forces from the corrected apt
                for i in range(nat):
                    config.atoms[i].frc = np.matmul(
                        np.transpose(pred_apt[0][i]), electric_field_vec
                    )

                if force_file:
                    write_conf(force_file, config.atoms, meta={'i': f'{itcount}'}, fmt='pf')

                for i in range(nat):
                    force[i] = config.atoms[i].frc

                if f_verbose:
                    print(" @DRIVER:   Forces computed.", flush=True)

            # broadcast force array from rank 0 to others
            force = comm.bcast(force, root=0)

            if rank == 0:
                f_data = True
                if f_verbose:
                    print(" @DRIVER:   f_data set to True after POSDATA", flush=True)

        # ---------------------------------------------------------------------
        # GETFORCE
        # ---------------------------------------------------------------------
        elif header == Message("GETFORCE"):
            if rank == 0:
                if f_verbose:
                    print(" @DRIVER:   Got GETFORCE request. Sending FORCEREADY", flush=True)
                sock.sendall(Message("FORCEREADY"))

                # sanity check
                if not isinstance(force, np.ndarray) or force.dtype != np.float64:
                    raise ValueError(
                        "driver returned forces with the wrong type or dtype. "
                        "Need a numpy.ndarray of 64-bit floats."
                    )

                if not isinstance(vir, np.ndarray) or vir.dtype != np.float64:
                    raise ValueError(
                        "driver returned virial with the wrong type or dtype. "
                        "Need a numpy.ndarray of 64-bit floats."
                    )

                if len(force.flatten()) != len(pos.flatten()):
                    raise ValueError(
                        "driver returned forces with the wrong size: number of "
                        "atoms and dimensions must match positions."
                    )

                if len(vir.flatten()) != 9:
                    raise ValueError(
                        "driver returned a virial tensor which does not have 9 components."
                    )

                extras = None

                if f_verbose:
                    print(" @DRIVER:   Sending pot, nat, force, vir, extras=0", flush=True)

                send_data(sock, np.float64(pot), f_verbose=f_verbose)
                send_data(sock, np.int32(nat), f_verbose=f_verbose)
                send_data(sock, force, f_verbose=f_verbose)
                send_data(sock, vir, f_verbose=f_verbose)
                send_data(sock, np.int32(0), f_verbose=f_verbose)

                f_data = False
                end_frc_send_time = time()
                if f_verbose:
                    print(
                        f" @DRIVER:   Iteration {itcount} took: {end_frc_send_time - start_pos_recv_time:.6f}s",
                        flush=True,
                    )
                itcount += 1

        # ---------------------------------------------------------------------
        # EXIT
        # ---------------------------------------------------------------------
        elif header == Message("EXIT"):
            if f_verbose:
                print(" @DRIVER:   Received EXIT message from i-PI. Closing down ...", flush=True)
            if rank == 0:
                if apt_file:
                    apt_file.close()
                if force_file:
                    force_file.close()
            return

        else:
            # If we have an unexpected header, print a message & possibly exit:
            if f_verbose:
                print(f" @DRIVER:   Received unknown header: {header}", flush=True)
            # You might want to handle it or break
            # For now, just continue
            continue

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

