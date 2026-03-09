#!/bin/bash
# Example SLURM batch script for running the ALE flow solver on a cluster.

#SBATCH --job-name=your_job_name
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=16 # Number of MPI processes
#SBATCH --nodes=1 # Number of nodes
#SBATCH -c 1
#SBATCH --partition=your_partition

# Load necessary modules
# This example shows how to do this with a conda environment
# created with the environment.yml file in this repository.
# Adjust as needed for your cluster setup.
module purge # Clear all loaded modules
module load miniconda3/py312/24.9.2-0 # Load the appropriate conda module for your cluster. Adjust the version as needed.
module load slurm # Load the Slurm module if not already loaded by default.

# Activate the conda environment
source /path_to_conda_dist/miniconda3/py312/24.9.2-0/bin/activate
conda activate /path_to_conda_env/.conda/envs/lagrangian-csf-env

# Avoid oversubscription of CPU cores by setting the number of OpenMP
# threads to match the allocated CPUs per task.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
CONDA_PREFIX=/path_to_conda_env/.conda/envs/lagrangian-csf-env
PROJECT_ROOT=/path_to_source_code/lagrangian-csf-ventricles/src

P=4 # Polynomial degree of the finite element space used in the linear elastodynamics simulations done to produce deformation field
E=1500 # Young's modulus of the linear elastodynamics simulations done to produce deformation field
K=2 # Bulk modulus of the linear elastodynamics simulations done to produce deformation field
DELTA_T=0.001 # Time step size of the linear elastodynamics simulations done to produce deformation field
T=4 # Total simulation time of linear elastodynamics simulations done to produce deformation field
N=2 # Number of cardiac cycles
EQUATIONS=navier-stokes # Type of equations to solve (navier-stokes or stokes)
MESH_SUFFIX=1 # Suffix for mesh files (0, 1 or 2)
VERSION=4 # Model version (see BDM_fluid_solver_ALE.py for details)
WRITE_OUTPUT=1 # Write output files?
GENERATE_CILIA_VECTORS=0 # Generate cilia direction vectors?

echo "Running flow simulations with $SLURM_NTASKS processes."
echo "Model version: $VERSION"
echo "Parameters: p=$P  E=$E  K=$K  dt=$DELTA_T  T=$T  eq=$EQUATIONS  mesh=$MESH_SUFFIX  cilia=$GENERATE_CILIA_VECTORS"

# Now run the flow solver using mpirun to launch the Python script across the allocated MPI processes.
$CONDA_PREFIX/bin/mpirun -n $SLURM_NTASKS $CONDA_PREFIX/bin/python \
$PROJECT_ROOT/BDM_fluid_solver_ALE.py -m $MESH_SUFFIX -p $P -s $E -T $T -dt $DELTA_T \
-n $N -k $K -g $EQUATIONS -c $WRITE_OUTPUT -o $WRITE_OUTPUT -v $VERSION -cd $GENERATE_CILIA_VECTORS

# Post-processing step (optional, can be run separately after the flow simulations are complete)
echo "Starting post processing."
SCRIPT=post_process_flow.py

$CONDA_PREFIX/bin/mpirun -n 1 $CONDA_PREFIX/bin/python $PROJECT_ROOT/$SCRIPT $T $K $P $VERSION $EQUATIONS $MESH_SUFFIX
