#!/bin/bash

#SBATCH --job-name=testOnly
#SBATCH --nodelist=a100
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output=testOnly_slurm-%j.out

source ../../pt_38_a100/bin/activate

# Your script goes here
date;
python3 detect.py -n convnext_tiny -w weights/convnext_tiny_noLM_sukarno_final.pth --image-path assets/Sukarno/ID-ANRI_FOTO_KEMPEN_RI_WILAYAH_JAKARTA_1955_24764.jpg --save-image
date;
