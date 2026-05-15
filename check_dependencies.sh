#!/bin/bash

#SBATCH --job-name=checkDep
#SBATCH --nodelist=a100
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output=checkDep_slurm-%j.out

# source ../../venv36b/bin/activate
source ../../pt_38_a100/bin/activate

# Your script goes here
date;
python3 check_dependencies.py
date;
