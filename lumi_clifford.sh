#!/bin/bash
#
#SBATCH --job-name="cliff"
#SBATCH --partition=small-g
#SBATCH --account project_462000921
#SBATCH -o test_clifford_9995_d4/clifford_circuit_test_less_exp_d4_400_1_over_h.out
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

SCRIPTTORUN="
cd /qhronos
python3 main_clifford.py --seed 1 --config clifford_circuit_test_less_exp_d4_400_steps --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
$EXEC bash -c "$SCRIPTTORUN"
echo $JOB_ID