#!/bin/bash

OMP_NUM_THREADS=64 mpirun -np 1 python3 train.py --inputfn train.xyz \
	--box 15.6627 0 0 0 15.6627 0 0 0 15.6627 \
	--committee_size 1 \
	--fixed_seed \
	--num_epochs 500 \
	> train.stdout 2> train.stderr

OMP_NUM_THREADS=64 python3 compare.py \
	--infile test.xyz \
	--box 15.6627 0 0 0 15.6627 0 0 0 15.6627 \
	--aptout apt.xyz \
	--scatterout scatter.dat \
	--rmseout rmse.dat 
