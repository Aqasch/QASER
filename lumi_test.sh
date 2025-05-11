#!/bin/bash
#
#SBATCH --job-name="testing"
#SBATCH --partition=small
#SBATCH --account project_462000520
#SBATCH -o test_plot.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:05:00
#
#
SCRIPTTORUN="
cd /qhronos
python3 plot_data.py
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"