#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p cpu
#SBATCH --qos="long-cpu"
#SBATCH --mem=24g
#SBATCH -o log/xraycpu.out


module load cuda/11.3



python3 scripts/coco_threatcombine2.py \
    --coco_annotation_path /home2/projects/datasets/CLCXray/annotation/CLCXray_train.json \
    --images_dir /home2/projects/datasets/CLCXray/image/train \
    --output_dir /home2/projects/datasets/CLCXray/SAM \
    --csv_path /home2/projects/datasets/CLCXray/SAM/CLCXray_train.csv




