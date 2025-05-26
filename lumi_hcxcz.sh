#!/bin/bash
#
#SBATCH --job-name="cliff"
#SBATCH --partition=small-g
#SBATCH --account project_462000921
#SBATCH -o test_HCXCZ/clifford_circuit_test_less_exp_dqnhcxcz_d4.out
#SBATCH --gpus=1
#SBATCH --mem=6G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

SCRIPTTORUN="
cd /qhronos
python3 main_hcxcz.py --seed 1 --config clifford_circuit_test_less_exp_dqnhcxcz_d4 --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
echo $JOB_ID
$EXEC bash -c "$SCRIPTTORUN"
