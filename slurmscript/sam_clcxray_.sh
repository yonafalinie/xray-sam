#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu
#SBATCH --job-name=clcxray
#SBATCH -o log/xray.out

module load cuda/11.3


python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/CLCXray/image/train \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_train.json \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_train.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/CLCXray/SAM/CLCXray_train_centroid.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/CLCXray/image/train \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_train.json \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_train.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/CLCXray/SAM/CLCXray_train_rand.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/CLCXray/image/train \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_train.json \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_train.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/CLCXray/SAM/CLCXray_train_bbox.json        

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/CLCXray/image/test \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_test.json \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_test.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/CLCXray/SAM/CLCXray_test_centroid.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/CLCXray/image/test \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_test.json \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_test.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/CLCXray/SAM/CLCXray_test_rand.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/CLCXray/image/test \
    --cocogt /home2/projects/datasets/CLCXray/annotation/CLCXray_test.json \
    --csvpath /home2/projects/datasets/CLCXray/SAM/CLCXray_test.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/CLCXray/SAM/CLCXray_test_bbox.json        

    