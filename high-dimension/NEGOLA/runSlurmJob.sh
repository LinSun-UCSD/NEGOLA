#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** MPI Job in Normal Queue ***
# 
# Last revised: 20 May 2019
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch runSlurmJob_lin_new.sh" on a Frontera login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per CLX node: 56
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J myjob           	# Job name
#SBATCH -o myjob.o%j       	# Name of stdout output file
#SBATCH -e myjob.e%j       	# Name of stderr error file
#SBATCH -p normal         	# Queue (partition) name
#SBATCH -N 10            	# Total # of nodes 
#SBATCH -n 110      	        # Total # of mpi tasks
#SBATCH -t 00:01:00        	# Run time (hh:mm:ss)
#SBATCH --mail-type=all    	# Send email at begin and end of job
#SBATCH -A ECS21015			# Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=lsun@ucsd.edu

srun hostname -s &> $(pwd)/slurmhosts.txt # Get node names


# Any other commands must follow all #SBATCH directives...
module list
pwd
date
ml python3
which python3
# ibrun python3 -m mpi4py.futures main.py
ibrun python3 -m mpi4py.futures -c "import pkg_resources; print([(d.project_name, d.version) for d in pkg_resources.working_set])"

    



