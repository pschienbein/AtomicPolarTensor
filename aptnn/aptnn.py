
import copy
import gc

from typing import Dict,Union
from enum import Enum, auto
from datetime import datetime

import numpy as np
import torch
from torch import cuda
from torch_geometric.data import DataLoader, Data

from e3nn.nn.models.v2106.gate_points_networks import SimpleNetwork

from aptnn.parameters import ModelParameters
from aptnn.box import Box


# TMP
def dump_gpu_memory_stats():
    gpus = list(range(cuda.device_count()))

    for i in gpus:
        print(i)
        print('alloc', cuda.memory_allocated(i) / 1024.0 / 1024.0)
        print('reserved', cuda.memory_reserved(i) / 1024.0 / 1024.0)
        print('max alloc', cuda.max_memory_allocated(i) / 1024.0 / 1024.0)
        print('max reserved', cuda.max_memory_reserved(i) / 1024.0 / 1024.0)
    print('', flush=True) 

#### TMP
#from mpi4py import MPI

class ELossFn(Enum):
    RMSE = auto()
    MAE = auto()

class ModifiedSimpleNetwork(SimpleNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.radial_cutoff = kwargs['max_radius']

    # Overwriting preprocess method of SimpleNetwork to adapt for our data
    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        edge_src = data['edge_index'][0]  # Edge source
        edge_dst = data['edge_index'][1]  # Edge destination 

        edge_vec = data['edge_vec']

        return batch, data['x'], edge_src, edge_dst, edge_vec



class APTNN:
    def __init__(self, device, model_parameters: ModelParameters, postfix=""):
        self.device = device
        self.model_parameters = copy.deepcopy(model_parameters)
        self.default_dtype = torch.float64
        self.postfix = postfix
        torch.set_default_dtype(self.default_dtype)

        # needed for restarts
        self.last_epoch = 0
        self.epoch_offset = 0

        # Create the network with the given parameters
        self.net = ModifiedSimpleNetwork(**self.model_parameters.to_kwargs())
        self.net.to(device)

        # Default Loss function
        self.loss_fn = ELossFn.RMSE

        # Default Optimizer
        self.learning_rate = 0.01
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # normalization
        self.norm_mean = dict()
        self.norm_std = dict()

        # initialize configuration index lists as empty
        self.train_data_indices = list()
        self.test_data_indices = list()

        # default training set fraction
        self.training_set_fraction = 0.9

        # NOTE hardcoded batch size to 1
        self.batch_size = 1

        # Hooks / handlers
        self.onTrainingEpochCallback = None

        self.edge_src = None

        # weight matrix of the offdiagonals
        self.offdiag_weight_matrix = (np.ones((3,3)) - np.identity(3)) * self.model_parameters.offdiag_weight + np.identity(3)
        self.offdiag_weight_matrix = self.offdiag_weight_matrix.flatten()

        # storage for unnormalize
        self.unnorm_prev_symbols = None


    ########################################
    #  Data Preprocessing
    ########################################

    def normalize(self, data):
        apts = dict()
        means = dict()
        stds = dict()

#        offdiag_weight = (np.ones((3,3)) - np.identity(3)) * self.offdiag_weight + np.identity(3)

#        print(offdiag_weight)


        for config in data:
            for atom in config.atoms:
                if atom.apt is not None:
                    if not atom.symbol in apts:
                        apts[atom.symbol] = []
                    apts[atom.symbol].append(atom.apt) 

        for symbol in apts:
            self.norm_mean[symbol] = np.mean(apts[symbol], axis=0)
            self.norm_std[symbol] = np.std(apts[symbol], axis=0)

#        print('', flush=True) 

        for config in data:
            for atom in config.atoms:
                if atom.apt is not None:
                    atom.apt = (atom.apt - self.norm_mean[atom.symbol]) / self.norm_std[atom.symbol]
        
        return data

    def create_graph_from_frame(self, frame):
        if frame.edge_src is None or frame.edge_dst is None or frame.edge_vec is None:
            cutoff2 = pow(self.model_parameters.radial_cutoff, 2.0)

            N = len(frame.atoms)
            size = N * N

            # create torch arrays if necessary
            bSetupBuffers = False
            if self.edge_src == None:
                bSetupBuffers = True
            elif len(self.edge_src) != size:
                bSetupBuffers = True

            if bSetupBuffers:
                #print('creating arrays')
                start = datetime.now()
                mAbs2Frac = frame.box.getAbs2FracMatrix()
                mFrac2Abs = frame.box.getFrac2AbsMatrix()
                t1 = []
                t2 = []
                l1 = []
                l2 = []
                for i in range(len(frame.atoms)):
                    for j in range(len(frame.atoms)):
                        t1.append(i)
                        t2.append(j)
                        l1.append(mAbs2Frac)
                        l2.append(mFrac2Abs)

                # these arrays are created once and forever; as long as the size of the input data are not changed...
                self.edge_src = torch.tensor(np.array(t1), dtype=torch.long, device=self.device)
                self.edge_dst = torch.tensor(np.array(t2), dtype=torch.long, device=self.device)
                self.lattice = torch.tensor(np.array(l1), dtype=self.default_dtype, device=self.device)
                self.latticeInv = torch.tensor(np.array(l2), dtype=self.default_dtype, device=self.device)
                #print('finished creating arrays:', datetime.now() - start)

            positions = []
            for atom in frame.atoms:
                positions.append(atom.position)
            positions = torch.tensor(np.array(positions), dtype=self.default_dtype, device=self.device)

            vd = positions[self.edge_dst[:]] - positions[self.edge_src[:]]

            # apply periodic boundary conditions:
            frac = torch.einsum('nij,ni->ni', self.lattice, vd)
            shift = torch.round(frac)
            vd -= torch.einsum('nij,ni->ni', self.latticeInv, shift)
            d = torch.einsum('ni,ni->n', vd, vd)

            mask = d < cutoff2

            # masked_select returns a deep copy according to the documentation
            frame.edge_src = torch.masked_select(self.edge_src, mask)
            frame.edge_dst = torch.masked_select(self.edge_dst, mask)
            frame.edge_vec = torch.masked_select(vd, mask.unsqueeze(-1).expand(vd.size())).reshape((len(frame.edge_src), 3))

            return frame

    ########################################

    def frame2TrainData(self, config):
        # Atom type array:
        types = []
        for atom in config.atoms:
            types.append(atom.symbol)
        N = len(types)
        #x_index = torch.tensor([self.model_parameters.atom_types.index(d[0]) for d in types])
        x_index = torch.tensor([self.model_parameters.atom_types.index(d) for d in types])
        x = torch.zeros(N, len(self.model_parameters.atom_types))
        x[range(N), x_index] = 1.0

        config = self.create_graph_from_frame(config)

#        positions = []
#        for iAtom in range(len(config.atoms)):
#            atom = config.atoms[iAtom]
#            positions.append(atom.position)

        # now the APTs, this is a bit tricky, because not all atoms in the given configurations have APTs defined...
        # this information needs to be stored for later
        apts = []
        positions = []
        aptWeight = []
        Nrelevant = 0
        for iAtom in range(len(config.atoms)):
            atom = config.atoms[iAtom]
            positions.append(atom.position)
            if atom.apt is not None:
                apts.append(np.einsum('ijk, jk->i', self.model_parameters.change_of_basis, atom.apt))
#                aptWeight.append(np.ones(9))
                aptWeight.append(self.offdiag_weight_matrix)
                Nrelevant += 1
            else:
                apts.append(np.zeros(9))
                aptWeight.append(np.zeros(9))

        y = torch.from_numpy(np.stack(apts, axis=0))

        return Data(
            pos=torch.tensor(np.array(positions)),
            x=x,
            edge_index=torch.stack([config.edge_src, config.edge_dst], dim=0),
            edge_vec=config.edge_vec,
            aptWeight=torch.tensor(np.array(aptWeight)),
            meanFactor=torch.tensor(len(positions) / Nrelevant),
            y=y
            )

    ########################################

    def frame2PredictData(self, config):
        # Atom type array:
        types = []
        for atom in config.atoms:
            types.append(atom.symbol)
        N = len(types)
        x_index = torch.tensor([self.model_parameters.atom_types.index(d) for d in types])
        x = torch.zeros(N, len(self.model_parameters.atom_types))
        x[range(N), x_index] = 1.0

        # create the graph representation
        config = self.create_graph_from_frame(config)

        positions = []
        for iAtom in range(len(config.atoms)):
            atom = config.atoms[iAtom]
            positions.append(atom.position)

        return Data(
            pos=torch.tensor(np.array(positions)),
            x=x,
            edge_index=torch.stack([config.edge_src, config.edge_dst], dim=0),
            edge_vec=config.edge_vec,
            )

    ########################################
    # Prediction
    ########################################

    def unnormalize(self, frame, predicted_normalized_apts):
        # use frame to get the atom symbols
        symbols = []
        for atom in frame.atoms:
            symbols.append(atom.symbol)
        print(f'frame symbols: {symbols}')
        print(f'prev symbols: {self.unnorm_prev_symbols}')
        print(f'norm_mean: {self.norm_mean}')
        print(f'norm_std: {self.norm_std}')
        
        # Update of the tensors needed?
        if self.unnorm_prev_symbols != symbols:
            # create the tensor used for unnormalizing this frame
            # This is only done when the symbols change, if it is a fixed trajectory, this initialization is done only once
            self.unnorm_mean_tensor = torch.tensor(np.array([self.norm_mean[d] for d in symbols]), device=self.device)
            self.unnorm_std_tensor = torch.tensor(np.array([self.norm_std[d] for d in symbols]), device=self.device)

        # store the symbol list
        self.unnorm_prev_symbols = symbols 

        return (predicted_normalized_apts * self.unnorm_std_tensor + self.unnorm_mean_tensor).cpu().detach().numpy()

    def predict_normal(self, data):
        tdata = list()
        for frame in data:
            tdata.append(self.frame2PredictData(frame))

        dataloader = DataLoader(tdata, 1)

        retdata = list()
        for ex in dataloader:
            ex = ex.to(self.device)
            y = self.net({'pos': ex.pos, 'x': ex.x, 'edge_index': ex.edge_index, 'edge_vec': ex.edge_vec, 'batch': ex.batch})
            y = torch.einsum('ijk,...i->...jk', self.model_parameters.change_of_basis, y) 
            retdata.append(y)

        return retdata

    # does the prediction and returns the unnormalized prediction as a numpy array
    def predict(self, data):
        retdata = self.predict_normal(data)
        for i in range(len(retdata)):
            retdata[i] = self.unnormalize(data[i], retdata[i])
        return retdata


    ########################################
    # Training
    ########################################

    def train(self, data, num_epochs):
        bRestart=False

        # get MSEs of all data and store them in per-symbol dictionary self.apt_ms
        tmpdata = dict()
        for frame in data: 
            for atom in frame.atoms: 
                if atom.apt is not None:
                    if atom.symbol not in tmpdata:
                        tmpdata[atom.symbol] = []
                    cob_apt = np.einsum('ijk, jk->i', self.model_parameters.change_of_basis, atom.apt)
                    tmpdata[atom.symbol].append(cob_apt**2)
                    

        self.apt_ms = dict()
        for symbol in tmpdata: 
            self.apt_ms[symbol] = np.mean(tmpdata[symbol], axis=0)

        
        # normalize data
        data = self.normalize(data)


        # split data into test/training
        # NOTE: test_data_indices may be empty!
        if len(self.train_data_indices) == 0:
            # This is not a restart of the training, generate the index lists now from the data set
            # 80% of the provided data set for training, the rest for testing!
            frac = 0.9
            ndata = len(data)
            
            self.train_data_indices = np.random.choice(range(ndata), int(frac*ndata), replace=False)
            self.test_data_indices = list(set(range(ndata)) - set(self.train_data_indices))
            
            # Again, training set might end up random...
            if len(self.train_data_indices) == 0:
                self.train_data_indices = [self.test_data_indices[0]]
                self.test_data_indices = self.test_data_indices[1:len(self.test_data_indices)]

            print("NOTE: This is a fresh start for a training, NOT a restart!")
        else:
            # Do nothing, just reuse the already set index lists, print a warning, though
            print("NOTE: Continuing training, you MUST ensure that the given data array is the very same as the first time the training started!")
            bRestart = True

         
#        print('Number of training data points: ', len(self.train_data_indices))
#        print('Training data indices: ', self.train_data_indices)
#        print('Number of validation data points: ', len(self.test_data_indices))
#        print('Validation data indices: ', self.test_data_indices)

        train_data = []
        for ndx in self.train_data_indices:
            train_data.append(self.frame2TrainData(data[ndx]))

        test_data = []
        for ndx in self.test_data_indices:
            test_data.append(self.frame2TrainData(data[ndx]))
            
        # Create an optimizer and a scheduler, if necessary

        # This will shuffle the data points at each epoch and the last batch is discarded if it contains less data 
        # points (total number of data points is not integer divisible by the batch size)
        train_dataloader = DataLoader(train_data, self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, self.batch_size)

        dataloaders = {
            "train": train_dataloader,
            "validation": test_dataloader
        }

        if bRestart:
            ofslog = open("learning_curve"+self.postfix+".dat", "a")
        else: 
            # If this is the first run, truncate the learning curve file an print out all indices used for training
            ofslog = open("learning_curve"+self.postfix+".dat", "w")

        ofslog.write('# Training structure indices:\n# ')
        for i in self.train_data_indices: 
            ofslog.write('%d '%i)
        ofslog.write('\n') 
        ofslog.write('# Test structure indices:\n# ')
        for i in self.test_data_indices: 
            ofslog.write('%d '%i)
        ofslog.write('\n') 


        ofslog.write('# Learning Curve:\n')
        ofslog.write('# Note that the RMSE / MAE live on the _normalized_ data scale\n')
        ofslog.write('# Epoch | RMSE (Train) | RMSE (Test) | MAE (Train) | MAE (Test) RMPE (Test) | Learning Rate\n')
        ofslog.flush()
        epoch = 0
        while epoch < num_epochs:
            ofslog.write("%d "%(epoch + self.epoch_offset))
            train_mse = 0.0
            train_mae = 0.0
            test_mse = 0.0
            test_mae = 0.0

            for phase in ['train', 'validation']:

                # skip validation phase if there is no data in the validation set
                if phase == 'validation' and len(self.test_data_indices) == 0:
                    continue

                running_mse = 0.0
                running_mae = 0.0
                norm = 0

#                ofslog.write('(')
                for ex in dataloaders[phase]:
                    ex = ex.to(self.device)
                    y = self.net({'pos': ex.pos, 'x': ex.x, 'edge_index': ex.edge_index, 'edge_vec': ex.edge_vec, 'batch': ex.batch})

                    diff = ((y - ex.y) * ex.aptWeight)
                    mse = (diff**2).mean() * ex.meanFactor
                    mae = diff.abs().mean() * ex.meanFactor

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        if self.loss_fn == ELossFn.MAE:
                            mae.backward()
                        elif self.loss_fn == ELossFn.RMSE:
                            mse.backward()
                        else:
                            raise RuntimeError("unknown loss function")
                        self.optimizer.step()

                    # NOTE: ex.x.size(0) equals to the number of atoms per frame times the batch size
                    # NOTE 2: The average is taken such that it is normed to a single atom, even if one configuration contains more APTs than others
                    running_mse += mse.detach() * ex.x.size(0)
                    running_mae += mae.detach() * ex.x.size(0)
                    norm += ex.x.size(0)

                epoch_mse = running_mse / norm
                epoch_mae = running_mae / norm

                if phase == 'validation':
                    test_mse = epoch_mse.item()
                    test_mae = epoch_mae.item()

                    self.scheduler.step(epoch_mse.item())
                else:
                    train_mse = epoch_mse.item()
                    train_mae = epoch_mae.item()

            ofslog.write('%e %e %e %e %e\n'%(np.sqrt(train_mse), np.sqrt(test_mse), train_mae, test_mae, self.optimizer.param_groups[0]['lr']))

            # call finished epoch callback if set
            if self.onTrainingEpochCallback != None:
                self.onTrainingEpochCallback(self)

            epoch += 1
            self.last_epoch = self.epoch_offset + epoch

            ofslog.flush()


    ########################################
    # I/O of the network
    ########################################


    def serialize(self, src = None):
        if src == None:
            savedata = dict()
            savedata['last_epoch'] = self.last_epoch
            savedata['learning_rate'] = self.learning_rate
            savedata['loss_fn'] = self.loss_fn 
            savedata['model'] = self.net.state_dict()
            savedata['model_parameters'] = self.model_parameters
            savedata['norm_mean'] = self.norm_mean
            savedata['norm_std'] = self.norm_std
            savedata['optimizer'] = self.optimizer.state_dict()
            savedata['scheduler'] = self.scheduler.state_dict()
            savedata['train_data_indices'] = self.train_data_indices
            savedata['test_data_indices'] = self.test_data_indices
            return savedata

        else:
            self.model_parameters = src['model_parameters']
            self.model_parameters.change_of_basis = self.model_parameters.change_of_basis.to(self.device, dtype=self.default_dtype)
            self.train_data_indices = src['train_data_indices']
            self.test_data_indices = src['test_data_indices']
            self.learning_rate = src['learning_rate']
            self.last_epoch = self.epoch_offset = src['last_epoch']

            # if for backward compatibility
            if 'loss_fn' in src:
                self.loss_fn = src['loss_fn']
            else:
                self.loss_fn = ELossFn.RMSE

            # recreate the net and load the data
            # necessary step if the model has been created with a different parameter set
            self.net = ModifiedSimpleNetwork(**self.model_parameters.to_kwargs())
            self.net.to(self.device)
            self.net.load_state_dict(src['model'])

            # Since the net was recreated, also the optimizer is restarted!
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(src['optimizer'])

            # reset the scheduler
            if 'scheduler' in src:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
                self.scheduler.load_state_dict(src['scheduler'])
            else:
                self.scheduler = None

            # normalization
            if 'norm_mean' in src:
                self.norm_mean = src['norm_mean']
            else:
                self.norm_mean = dict()
                for kind in self.model_parameters.atom_types:
                    self.norm_mean[kind] = np.zeros((3,3))
                print('# NOTE: could not find normalization mean in input file, setting to', self.norm_mean)
            
            if 'norm_std' in src:
                self.norm_std = src['norm_std']
            else:
                self.norm_std = dict()
                for kind in self.model_parameters.atom_types:
                    self.norm_std[kind] = np.ones((3,3))
                print('# NOTE: could not find normalization stddev in input file, setting to', self.norm_std)

    def load(self, sFn):
        try:
            loaded_data = torch.load(sFn)
        except:
            # The file could not be loaded! abort the whole process!
            print("Data file", sFn, "could not be loaded! Aborting process!")
            exit()

        self.serialize(loaded_data) 

    def save(self, sFn):
        savedata = self.serialize()
        torch.save(savedata, sFn)            



#test = APTNN(device='cpu', model_parameters=ModelParameters(atom_types=['O', 'H']))



