#!/bin/sh
cd /esat/izar/r0833114/OAA
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss
EXP=exp2
# train an AffinityNet as a segmentation network
/esat/christianso/r0833114/miniconda3/envs/wsss/bin/python ./scripts/train_aff.py \
            --data_root "/esat/izar/r0833114/VOCdevkit/VOC2012" \
            --weights "/esat/izar/r0833114/OAA/pretrained_model/ilsvrc_cls.params" \
            --la_crf_dir "results_voc/${EXP}/results_crf_4.0" \
            --ha_crf_dir "results_voc/${EXP}/results_crf_24.0" \
            --session_name "resnet38_exp2" \
            --snapshot_dir "checkpoints/train_aff/${EXP}/" 