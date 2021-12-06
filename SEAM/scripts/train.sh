#!/bin/sh
cd /home/charis/kul-thesis/SEAM

python ./scripts/train.py \
    --data_root "/home/charis/kul-thesis/kvasir-dataset-v2-new/" \
    --train_list "./kvasirv2/train.txt" \
    --val_list "./kvasirv2/val.txt" \
    --weights "/home/charis/kul-thesis/ilsvrc_cls.params" \
    --session_name "resnet38_kvasir"
