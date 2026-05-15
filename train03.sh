#!/bin/bash

#SBATCH --job-name=Sukarno_ConvNext_tiny_noLM_RetinaFace
#SBATCH --nodelist=a100
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output=train_cnTiny_sukarno_noLM_slurm-%j.out

source ../../pt_38_a100/bin/activate

# Your script goes here
date;
python3 train.py --network convnext_tiny --train-data ./data/sukarno/train --batch-size 4
date;
