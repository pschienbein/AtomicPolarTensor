
mpirun -np 1 python3 train.py --inputfn input.xyz \
	--box 15.6627 0 0 0 15.6627 0 0 0 15.6627 \
	--committee_size 1 \
	--fixed_seed \
	> train.stdout 2> train.stderr

# using a previously trained aptnn on input.xyz should always yield the exakt same result
mpirun -np 1 python3 predict.py --trajectory input.xyz \
	--aptnn aptnn.torch.ref \
	--aptout apt-from-aptnn.torch.ref.xyz \
	--box 15.6627 0 0 0 15.6627 0 0 0 15.6627 \
	> predict1.stdout 2> predict1.stderr


# using a freshly trained aptnn (first command) on input.xyz always yields somewhat different APTs because of the random initialization of the network
mpirun -np 1 python3 predict.py --trajectory input.xyz \
	--box 15.6627 0 0 0 15.6627 0 0 0 15.6627 \
	> predict2.stdout 2> predict2.stderr




