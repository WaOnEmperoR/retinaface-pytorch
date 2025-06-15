#!/bin/bash

#SBATCH --job-name=train_rf_resnet34
#SBATCH --nodelist=a100
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1

source ../../pt_38_a100/bin/activate

# Your script goes here
date;
python3 train.py --network resnet34
date;
