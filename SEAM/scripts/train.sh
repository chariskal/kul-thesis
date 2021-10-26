#!/bin/sh
cd /esat/izar/r0833114/SEAM
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss

/esat/christianso/r0833114/miniconda3/envs/wsss/bin/python ./scripts/train.py \
    --data_root "/esat/izar/r0833114/kvasir_v2" \
    --train_list "./kvasirv2/train.txt" \
    --val_list "./kvasirv2/val.txt" \
    --weights "./pretrained_model/ilsvrc_cls.params" \
    --session_name "resnet38_kvasir"
