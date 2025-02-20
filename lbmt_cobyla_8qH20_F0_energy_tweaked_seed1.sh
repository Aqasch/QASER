#!/bin/bash
#
#SBATCH --job-name="qhronos"
#SBATCH --partition=gpu
#SBATCH --clusters ukko
#SBATCH -o lbmt_cobyla_8qH20_F0_energy_tweaked_seed1.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH -G 1
#SBATCH --priority=5100000
#SBATCH --begin=2024-06-13T17:18:00
#
#
srun python main.py --seed 1 --config lbmt_cobyla_8qH2O_step_250_F0_energy_tweaked --experiment_name "finalize/"