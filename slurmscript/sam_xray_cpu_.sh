#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p cpu
#SBATCH --qos="long-cpu"
#SBATCH --mem=24g
#SBATCH -o log/xraycpu.out


module load cuda/11.3



python3 scripts/coco2csv.py \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_test.json \
    --image /home2/projects/datasets/CLCXray/image/test \
    --output /home2/projects/datasets/CLCXray/SAM \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_test.csv




