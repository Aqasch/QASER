#!/bin/bash
#
#SBATCH --job-name="clifford_circuit_test_less_exp_d4_rand_cut5_seed1"
#SBATCH --partition=small-g
#SBATCH --account project_462000921
#SBATCH -o test_warm_start/clifford_circuit_test_less_exp_d4_rand_cut5_seed1.out
#SBATCH --gpus=1
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00

SCRIPTTORUN="
cd /qhronos
python3 main_warm_start_random.py --seed 1 --config clifford_circuit_test_less_exp_d4_rand_cut5 --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
echo $JOB_ID
$EXEC bash -c "$SCRIPTTORUN"


