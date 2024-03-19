

import numpy as np
import torch
from torch_geometric.data import DataLoader, Data

import gc
import sys
import time
import logging
import socket
from datetime import datetime

from mpi4py import MPI

from e3nn.o3 import ReducedTensorProducts

from aptnn.aptnn import APTNN
from aptnn.parameters import ModelParameters

from aptnn.box import Box

# TMP
def dump_gpu_memory_stats():
    from torch import cuda
    gpus = list(range(cuda.device_count()))

    for i in gpus:
        print(i)
        print('alloc', cuda.memory_allocated(i) / 1024.0 / 1024.0)
        print('reserved', cuda.memory_reserved(i) / 1024.0 / 1024.0)
        print('max alloc', cuda.max_memory_allocated(i) / 1024.0 / 1024.0)
        print('max reserved', cuda.max_memory_reserved(i) / 1024.0 / 1024.0)
    print('', flush=True) 

# This committee NNP class trivially parallelizes the problem such that each committee member gets its own process.
# Through MPI the processes can reside on different machines.
class CommitteeAPTNN:
    def __init__(self, committee_size, model_parameters, device_map=None):
        self.model_parameters = model_parameters

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()       

        # if no initialization is given, provide dummy values which will be overwritten later when load is called
        if committee_size is None and self.model_parameters is None:
            committee_size = comm.Get_size()
            self.model_parameters = ModelParameters(atom_types = ['H'])

        if comm.Get_size() != committee_size:
            raise RuntimeError("Number of MPI Processes should match the committee size!")

        if device_map is None:
            # assign gpu based on the assumption that the number of GPUs is the same on all hosts
            bGPU = torch.cuda.is_available()
            if bGPU:
                nGPU = torch.cuda.device_count()
                sDevice = 'cuda:%d'%(self.rank % nGPU)
                print('rank %d located on %s mapped to GPU %s'%(self.rank, socket.gethostname(), sDevice), end=', ')
            else:
                sDevice = 'cpu'
                print('rank %d located on %s mapped to CPU'%(self.rank, socket.gethostname()), end=', ')
        else:
            sDevice = device_map[self.rank]
            print('rank %d, using device_map, assigning %s'%(self.rank, sDevice), end=', ')

        self.device = torch.device(sDevice)
        print('using %d threads'%(torch.get_num_threads()), flush=True)

        # also set the default device, in this implementation one MPI process equals to one committee member, which lives on one device only
        #torch.cuda.device(self.device)

        # construct the APTNN
        self.rank_member = APTNN(device=self.device, model_parameters=self.model_parameters, postfix="_"+str(self.rank))

        # override the callback function
        self.rank_member.onTrainingEpochCallback = self.onTrainingEpochCallbackCommitteeWrapper
        self.onTrainingEpochCallback = None

    def onTrainingEpochCallbackCommitteeWrapper(self,net):
        if self.onTrainingEpochCallback != None:
            self.onTrainingEpochCallback(self)


    def predict_loop(self):
        comm = MPI.COMM_WORLD
        if comm.rank > 0:
            while True:
                if self.predict(None) > 0:
                    return


        ####################

    # Each MPI rank does the training only for its assigned committee member. 
    # Using num_active_processes can be used to reduce the number of trained committee members in parallel; 
    # this might be necessary to avoid out of memory errors
    # having 8 members in total and using num_active_processes = 4, 
    # ensures that only 4 out of 8 processes run at the same time, thus reducing the amount of memory required, 
    # however, the calculation clearly takes longer!
    def serializeMPI(self, fnWork, num_active_processes):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # no serialization necessary
        if num_active_processes == 0 or num_active_processes >= comm.Get_size():
            return

        # number of committee members
        num_tasks = comm.Get_size()

        # assemble a list where each process is working on to rank 0
        devices = comm.gather(str(self.device), root=0)

        # rank 0 has the role of the director who receives a note when another rank is finished
        # and then releases the next one, finally it releases itself.
        if rank == 0:
            # create a device list to distribute the processed to available resources correctly
            # note: rank 0 not in that list!
            dev2rank = {}
            running = {}
            for i in range(1,len(devices)):
                dev = devices[i]
                if dev not in dev2rank:
                    dev2rank[dev] = []
                    running[dev] = 0
                dev2rank[dev].append(i)
            
            # how many processes per device 
            num_procs_per_device = int(num_active_processes / len(dev2rank))
            if num_procs_per_device % 2 != 0:
                print('WARNING: Requested total active processes', num_active_processes, 'on', len(dev2rank), 'different devices: Not the same number of processes will be running on each device!')

            # now release tasks accordingly
            for device in dev2rank:
                for i in range(num_procs_per_device):
                    # release the last process
                    comm.isend(0, dev2rank[device][-1])
                    dev2rank[device].pop()
                    running[device] += 1


            todo = num_tasks - num_active_processes - 1
            while todo > 0 or running[str(self.device)] == num_procs_per_device:
                #LAST ONE!!
                # If todo ==0 AND  N(cuda:0) < 2, then release rank 0!!!
                fin_dev = comm.recv()
                running[fin_dev] -= 1

                if len(dev2rank[fin_dev]) == 0:
                    continue

                comm.isend(0, dev2rank[fin_dev][-1])
                running[fin_dev] += 1

                dev2rank[fin_dev].pop()

                todo -= 1
            

        else: # rank != 0
            comm.recv()

        # do the work
        ret = fnWork()

        # free unused training memory
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

        # report finished
        if rank >= num_active_processes:
            comm.isend(str(self.device), 0)

        return ret

        ####################

    # does the prediction for the committee; 
    # NOTE the returned predicted apt is "unnormalized", i.e. in absolute units and thus independent from the trained networks;
    #      the returned standard deviation is in the normalized scale of the network, dependent on the network!
    def predict(self, data, num_active_processes = 0):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        start = datetime.now()

        # Broadcast number of active processes from rank 0
        num_active_processes = comm.bcast(num_active_processes, root=0)

        # Broadcast the data from rank 0
        data = comm.bcast(data, root=0)

        if data == None:
            if rank == 0:
                return None
            else:
                return False

        net = self.rank_member
        if num_active_processes > 0:
            prediction = self.serializeMPI(lambda: net.predict_normal(data), num_active_processes)
        else:
            prediction = net.predict_normal(data)

        prediction_unnormal = []
        for i in range(len(prediction)):
            prediction_unnormal.append(net.unnormalize(data[i], prediction[i]))
            prediction[i] = prediction[i].cpu().detach().numpy()

        # note: to perform gather, all tensors must be transported to the CPU. At this stage it is not reasonable to use torch anylonger...
        # -> collection of the data via numpy
        gatherdata = [ prediction, prediction_unnormal ]
        gatherdata = comm.gather(gatherdata, root=0)

        # compute averages on the returned data
        retdata = { 'apt': list(), 'std': list() }
        if rank == 0:
            gatherdata = np.swapaxes(gatherdata, 0, 1)
            prediction_normal = np.swapaxes(gatherdata[0], 0, 1)
            prediction_unnormal = np.swapaxes(gatherdata[1], 0, 1)

            for i in range(len(prediction_normal)):
                retdata['apt'].append(np.mean(prediction_unnormal[i], axis=0))
                retdata['std'].append(np.std(prediction_normal[i], axis=0))

        if rank == 0:
            return retdata
        else:
            return True


    def train(self, data, num_epochs, num_active_processes = 0):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        net = self.rank_member
        if num_active_processes > 0:
            self.serializeMPI(lambda: net.train(data, num_epochs), num_active_processes)
        else:
            net.train(data, num_epochs)

        # wait for all processes to be finished with their trainings
        comm.barrier() 

        pass


    def load(self, sFn):
        comm = MPI.COMM_WORLD

        loaded_data = None
        if self.rank == 0:
            try:
                if  torch.cuda.is_available():
                    loaded_data = torch.load(sFn)
                else:
                    loaded_data=torch.load(sFn,map_location=self.device)
                if len(loaded_data) != comm.Get_size():
                    print("Number of MPI Processes must match the committee size", file=sys.stderr)
                    comm.Abort(1)

            except Exception as e:
                # The file could not be loaded! abort the whole MPI process!
                print("Data file", sFn, "could not be loaded! Aborting MPI process!", file=sys.stderr)
                print(f"the exception is: {e}.",file=sys.stderr)
                comm.Abort(1)

        # scatter member data        
        loaded_data = comm.scatter(loaded_data, root=0)
        self.rank_member.serialize(loaded_data)

     
    def save(self, sFn):
        comm = MPI.COMM_WORLD
        
        savedata = self.rank_member.serialize()
        savedata = comm.gather(savedata, root=0)

        if comm.Get_rank() == 0:
            torch.save(savedata, sFn)

