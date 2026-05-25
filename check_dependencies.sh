#!/bin/bash

#SBATCH --job-name=checkDep
#SBATCH --nodelist=a1
#SBATCH --partition=short
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=24G
#SBATCH --output=checkDep_slurm-%j.out

source ../../venv36b/bin/activate
# source ../../pt_38_a100/bin/activate
# source ../../waone_310/bin/activate

# Your script goes here
date;
python3 check_dependencies.py
date;
