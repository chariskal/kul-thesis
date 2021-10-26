#!/bin/sh
EXP=exp2

cd /esat/izar/r0833114/OAA
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss

/esat/christianso/r0833114/miniconda3/envs/wsss/bin/python ./scripts/infer.py \
    --img_dir=/esat/izar/r0833114/VOCdevkit/VOC2012/JPEGImages/ \
    --infer_list=./voc12/val_oaa.txt \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=256 \
	--num_classes=20 \
    --restore_from=./checkpoints/train/${EXP}/pascal_voc_epoch_14.pth \
    --save_dir=/esat/izar/r0833114/OAA/results_voc/${EXP}/attention/ \
    --out_cam=/esat/izar/r0833114/OAA/results_voc/${EXP}/results_cam \
    --out_crf=/esat/izar/r0833114/OAA/results_voc/${EXP}/results_crf \
