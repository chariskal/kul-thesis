#!/bin/sh
cd /home/charis/kul-thesis/OAA
EXP=exp9

python ./scripts/train.py \
    --data_root='/home/charis/kul-thesis/kvasir-dataset-v2-new/' \
    --train_list='kvasirv2/train_new.txt' \
    --test_list='kvasirv2/val_new.txt' \
    --epoch=14 \
    --lr=0.001 \
    --batch_size=8 \
    --dataset=kvasir \
    --input_size=256 \
    --disp_interval=100 \
    --num_classes=8 \
    --num_workers=8 \
    --snapshot_dir=checkpoints/train/${EXP}/ \
    --att_dir=./results_kvasir/${EXP}/attention/ \
    --decay_points='5,10'
