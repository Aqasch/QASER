#!/bin/bash
#
#SBATCH --job-name="e8h20"
#SBATCH --partition=small-g
#SBATCH --account project_462000520
#SBATCH -o noisy_8qH2O_step_250_seed10.out
#SBATCH --gpus=1
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00


SCRIPTTORUN="
cd /qhronos
python3 main_noisy.py --seed 10 --config lbmt_cobyla_8qH2O_step_250 --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"