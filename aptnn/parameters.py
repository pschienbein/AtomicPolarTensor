
from enum import Enum, auto
from dataclasses import dataclass, field

import torch

from e3nn.o3 import ReducedTensorProducts

@dataclass
class ModelParameters:
    atom_types: list 
    irreps_in: str = field(init=False) # irreps_in is generated later...
    irreps_out: str = field(default="1x0e+1x1e+1x2e", init=False)
    radial_cutoff: float = 6.0
    num_neighbors: float = 90.0
    num_nodes: float = 5.0
    pool_nodes: bool = False
    lmax: int = 2
    num_features: int = 50
    num_layers: int = 3
    change_of_basis: torch.tensor = field(init=False)
    offdiag_weight: float = 1.0

    def __post_init__(self) -> None:
        self.irreps_in = str(len(self.atom_types)) + "x0e"
        rtp = ReducedTensorProducts('ij', i='1o', j='1o')
        self.change_of_basis = rtp.change_of_basis

    def to_kwargs(self):
        return dict(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_out,
                    max_radius=self.radial_cutoff, 
                    num_neighbors=self.num_neighbors,
                    num_nodes=self.num_nodes,
                    pool_nodes=self.pool_nodes,
                    mul=self.num_features,
                    layers=self.num_layers,
                    lmax=self.lmax)

