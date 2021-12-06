#!/bin/sh
EXP=exp9

cd /home/charis/kul-thesis/OAA

python ./scripts/infer.py \
    --img_dir=/home/charis/kul-thesis/kvasir-dataset-v2-new/test \
    --infer_list=/home/charis/kul-thesis/OAA/kvasirv2/val_new.txt \
    --batch_size=1 \
    --dataset=kvasir \
    --input_size=256 \
	--num_classes=8 \
    --restore_from=./checkpoints/train/${EXP}/kvasirv2_epoch_15.pth \
    --save_dir=./results_kvasir/${EXP}/attention/ \
    --out_cam=./results_kvasir/${EXP}/results_cam \
    --out_crf=./results_kvasir/${EXP}/results_crf \
