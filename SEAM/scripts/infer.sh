#!/bin/sh
cd /esat/izar/r0833114/SEAM
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss

/esat/christianso/r0833114/miniconda3/envs/wsss/bin/python ./scripts/infer.py \
    --weights "/esat/izar/r0833114/SEAM/pretrained_model/resnet38_kvasir.pth" \
    --data_root "/esat/izar/r0833114/kvasir_v2" \
    --out_cam "./results_kvasir/results_cam" \
    --out_crf "./results_kvasir/results_crf" \
    --out_cam_pred "./results_kvasir/results_cam_pred" \
    --infer_list "kvasirv2/train.txt" \
    --num_classes 8