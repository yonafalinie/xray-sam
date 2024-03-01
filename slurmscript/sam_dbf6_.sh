#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu
#SBATCH --job-name=dbxray
#SBATCH -o log/dbxray.out

module load cuda/11.3


python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/dbf6/images \
    --cocogt /home2/projects/datasets/dbf6/annotations/dbf6_train__.json \
    --csvpath /home2/projects/datasets/dbf6/SAM/dbf6_train__.csv  \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/dbf6/SAM/dbf6_train__centroid.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/dbf6/images \
    --cocogt /home2/projects/datasets/dbf6/annotations/dbf6_train__.json \
    --csvpath /home2/projects/datasets/dbf6/SAM/dbf6_train__.csv  \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/dbf6/SAM/dbf6_train__rand.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/dbf6/images \
    --cocogt /home2/projects/datasets/dbf6/annotations/dbf6_train__.json \
    --csvpath /home2/projects/datasets/dbf6/SAM/dbf6_train__.csv  \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/dbf6/SAM/dbf6_train__bbox.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/dbf6/images \
    --cocogt /home2/projects/datasets/dbf6/annotations/dbf6_test.json \
    --csvpath /home2/projects/datasets/dbf6/SAM/dbf6_test.csv  \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/dbf6/SAM/dbf6_test_centroid.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/dbf6/images \
    --cocogt /home2/projects/datasets/dbf6/annotations/dbf6_test.json \
    --csvpath /home2/projects/datasets/dbf6/SAM/dbf6_test.csv  \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/dbf6/SAM/dbf6_test_rand.json        

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/dbf6/images \
    --cocogt /home2/projects/datasets/dbf6/annotations/dbf6_test.json \
    --csvpath /home2/projects/datasets/dbf6/SAM/dbf6_test.csv  \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/dbf6/SAM/dbf6_test_bbox.json         