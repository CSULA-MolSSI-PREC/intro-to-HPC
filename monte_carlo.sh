#!/bin/bash

# Set the number of cores and processes per node
#SBATCH --ntasks-per-node=16

# Account to be charged for resources used
#SBATCH --account=bbow-delta-gpu

# Select GPU partition
#SBATCH --partition=gpuA40x4 

# Set the walltime
#SBATCH --time=00:15:00

# Set the output file name
#SBATCH --output=monte_carlo.out

# Set the error file name
#SBATCH --error=monte_carlo.err

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,per_task:1

# Load the necessary modules
module load python

# Run the python script in parallel using mpirun
mpirun -np $SLURM_NTASKS python monte_carlo_pi_parallel.py
