#!/bin/sh
cd /esat/izar/r0833114/SEAM
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss
# train an AffinityNet as a segmentation network
/esat/christianso/r0833114/miniconda3/envs/wsss/bin/python ./scripts/train_aff.py \
        --voc12_root "/esat/izar/r0833114/VOCdevkit/VOC2012" \
        --weights "/esat/izar/r0833114/SEAM/pretrained_model/ilsvrc_cls.params" \
        --la_crf_dir "results_voc/results_crf_4.0" \
        --ha_crf_dir "results_voc/results_crf_24.0" \
        --session_name "resnet38_exp1"\