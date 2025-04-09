#!/bin/bash
#
#SBATCH --job-name="testing"
#SBATCH --partition=small-g
#SBATCH --account project_462000520
#SBATCH -o analysis_lbmt_cobyla_6qLiH_step_100_F0_energy_depth_up_seed0.out
##SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:05:00
#
#
SCRIPTTORUN="
cd /qhronos
python3 data_analysis.py
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"