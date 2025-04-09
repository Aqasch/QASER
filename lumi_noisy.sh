#!/bin/bash
#
#SBATCH --job-name="n_lih"
#SBATCH --partition=small-g
#SBATCH --account project_462000520
#SBATCH -o noisy_6qLiH_step_100_F0_energy_untweaked_seed0.out
#SBATCH --gpus=1
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00


SCRIPTTORUN="
cd /qhronos
python3 main_noisy.py --seed 0 --config lbmt_cobyla_6qLiH_step_100_F0_energy_untweaked --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"