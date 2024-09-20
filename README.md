Training and prediction of atomic polar tensors using the e3nn framework implementing E(3) equivariant neural network on top of torch.
Note that the code is working and tested on liquid ambient water (see publication below), but still in beta. It is therefore not optimized and might be subject to change in the future. 

# Installation 

Download and add basedir to the PYTHONPATH environment variable and basedir/scripts to the PATH environment variable.

**Requires** e3nn (https://github.com/e3nn/e3nn) and pytorch.
Tested with 
- python 3.10.7
- e3nn 0.5.0
- torch 1.12.1+cu116

# Finite Electric Field Simulations
All Data and Code necessary to run finite E-Field simulations can be found here: 
https://github.com/kjaj98/cp2k-apt-pnnp-paper

# Related Publications

```
@Article{APTNN,
  author    = {Schienbein, Philipp},
  journal   = {J. Chem. Theory Comput.}, 
  title     = {Spectroscopy from Machine Learning by Accurately Representing the Atomic Polar Tensor},
  year      = {2023},
  issn      = {1549-9618},
  pages     = {705--712},
  volume    = {19}, 
  doi       = {10.1021/acs.jctc.2c00788},
}

@Article{APTNN-EField,
  author  = {Joll, K. and Schienbein, P. and Rosso, K. M. and Blumberger, J.}, 
  journal = {Nat. Commun.}, 
  title   = {Molecular dynamics simulation with finite electric fields using Perturbed Neural Network Potentials}, 
  year    = {2024}, 
  pages   = {8192}, 
  volume  = {15}, 
  doi     = {10.1038/s41467-024-52491-3}, 
}

```


