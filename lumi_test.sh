#!/bin/bash
#
#SBATCH --job-name="testing"
#SBATCH --partition=standard-g
#SBATCH --account project_462000520
#SBATCH -o lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked_seed5_analysis.out
#SBATCH --nodes=1
##SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:01:00
#
#
SCRIPTTORUN="
cd /qhronos
python3 data_analysis.py lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked 5
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"