#!/bin/bash
#
#SBATCH --job-name="cz"
#SBATCH --partition=small-g
#SBATCH --account project_462000921
#SBATCH -o test_CZ/graph_9_1_60_99995_500.out
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

SCRIPTTORUN="
cd /qhronos
python3 main_cz.py --seed 1 --config graph_9_1_60_99995_500 --experiment_name \"finalize/\"
"

export EXEC="srun singularity exec -B $(pwd):/qhronos apptainer/images/qhronos.sif"
echo $JOB_ID
$EXEC bash -c "$SCRIPTTORUN"
