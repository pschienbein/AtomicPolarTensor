
from dataclasses import dataclass, field
from typing import List
import numpy as np

from aptnn.box import Box

@dataclass
class Atom:
    symbol: str
    position: np.ndarray
    velocity: np.ndarray = field(default=None)
    apt: np.ndarray = field(default=None)
    apt_total_std_norm: float = field(default=None)
    apt_total_std_unnorm: float = field(default=None)
    apt_std: np.ndarray = field(default=None)
    frc: np.ndarray = field(default=None)



@dataclass
class Frame:
    atoms: list 
    frameno: int = field(default = 0)
    meta: dict = field(default = dict)
    box: Box = field(default = None)
    edge_src: np.ndarray = field(default = None)
    edge_dst: np.ndarray = field(default = None)
    edge_vec: np.ndarray = field(default = None)

