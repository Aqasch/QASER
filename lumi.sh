#!/bin/bash
#
#SBATCH --job-name="o_lih"
#SBATCH --partition=small-g
#SBATCH --account project_462000520
#SBATCH -o no_noise_lbmt_cobyla_6qLiH_step_100_seed10.out
#SBATCH --gpus=1
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00


SCRIPTTORUN="
cd /qhronos
python3 main.py --seed 10 --config lbmt_cobyla_6qLiH_step_100 --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"