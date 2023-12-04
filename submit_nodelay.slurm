#!/bin/bash -l
#SBATCH --job-name=Flow_NoDelay
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --partition=guest_gpu
#SBATCH --output=/work/cseos2g/papapalpi/logs/%x-%j.out

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd $COMMON/code/cs6376-project-2023
srun singularity exec --nv $COMMON/flow-desktop_latest.sif ./train_bash.bash ring
