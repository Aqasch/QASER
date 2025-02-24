#!/bin/bash
#
#SBATCH --job-name="qhronos"
#SBATCH --partition=standard-g
#SBATCH --account project_462000520
#SBATCH -o lbmt_cobyla_8qH20_F0_energy_untweaked_seed3.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#
#
srun singularity exec apptainer/images/qhronos.sif python3 main.py --seed 3 --config lbmt_cobyla_8qH2O_step_250_F0_energy_untweaked --experiment_name "finalize/"