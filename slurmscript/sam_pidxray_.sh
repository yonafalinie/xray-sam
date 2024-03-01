#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH -t 07-00:00:00
#SBATCH --mem=24g
#SBATCH --gres=gpu
#SBATCH --job-name=pidxray
#SBATCH -o log/pidxray.out

module load cuda/11.3


python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/train \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_train.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_train.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_train_centroid.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/train \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_train.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_train.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_train_rand.json

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/train \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_train.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_train.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_train_bbox.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/easy \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_easy.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_easy.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/pidray/SAM/ppidray_test_easy_rand.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/easy \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_easy.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_easy.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_easy_rand.json     

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/easy \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_easy.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_easy.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_easy_bbox.json               

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/hard \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_hard.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_hard.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_hard_centroid.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/hard \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_hard.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_hard.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_hard_rand.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/hard \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_hard.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_hard.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_hard_bbox.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/hidden \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_hidden.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_hidden.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_centroid \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_hidden_centroid.json    

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/hidden \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_hidden.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_hidden.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_rand \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_hidden_rand.json     

python3 scripts/sam_prompt.py \
    --image /home2/projects/datasets/pidray/image/test/hidden \
    --cocogt /home2/projects/datasets/pidray/annotation/pidray_test_hidden.json \
    --csvpath /home2/projects/datasets/pidray/SAM/pidray_test_hidden.csv \
    --weight /home3/qljx17/SAM/xray-sam/weight/sam_vit_h_4b8939.pth \
    --prompt_bbox \
    --output_json /home2/projects/datasets/pidray/SAM/pidray_test_hidden_bbox.json        