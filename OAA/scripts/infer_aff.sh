#!/bin/sh
cd /esat/izar/r0833114/SEAM
source /esat/izar/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss
# infer about AffinityNet
/esat/izar/r0833114/miniconda3/envs/wsss/bin/python infer_aff.py --weights "resnet38_exp2.pth" --infer_list "voc12/val.txt" --cam_dir "results_cam" --voc12_root "/esat/izar/r0833114/VOCdevkit/VOC2012" --out_rw "results_seg"