#!/bin/bash
#SBATCH -p GPU
#Requesting one node
#SBATCH -N1
#SBATCH -n25
#Requesting 1 A100 GPU
#SBATCH --gres=gpu:a100:1

module purge
module load openmpi/4.1.0
module load gcc/8.3.0
module load cp2k/2024.3

#filter warnings
export OMPI_MCA_op=^avx
export OMPI_MCA_btl="^openib"
export OMPI_MCA_orte_base_help_aggregate=0

# Initialize Conda
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh

# Activate Conda environment
conda activate /home/kjoll/apt_conda_env

# Set PYTHONPATH
export PYTHONPATH=/home/kjoll/public_APT:$PYTHONPATH

# Start i-PI server
OMP_NUM_THREADS=1 mpirun -np 17 cp2k.psmp -i start.inp -o apt_ipi_test.out &

# Wait for the server to start
sleep 5

# Run the driver with MPI
mpirun -np 8 /home/kjoll/public_APT/scripts/i-pi-driver.py -u -a cp2k -i driver.inc > driver.out 2>&1

wait

