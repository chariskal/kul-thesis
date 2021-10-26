#!/bin/sh
cd /esat/izar/r0833114/OAA
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss
EXP=exp1

/esat/christianso/r0833114/miniconda3/envs/wsss/bin/python ./scripts/train.py \
    --data_root='/esat/izar/r0833114/kvasir_v2/' \
    --train_list='kvasirv2/train.txt' \
    --test_list='kvasirv2/val.txt' \
    --epoch=14 \
    --lr=0.001 \
    --batch_size=4 \
    --dataset=kvasir \
    --input_size=256 \
    --disp_interval=100 \
    --num_classes=8 \
    --num_workers=8 \
    --snapshot_dir=checkpoints/train/${EXP}/ \
    --att_dir=./results_kvasir/${EXP}/attention/ \
    --decay_points='5,10'
