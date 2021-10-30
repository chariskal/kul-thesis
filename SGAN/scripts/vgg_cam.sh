#!/bin/bash
cd /home/charis/kul-thesis/SGAN
source /home/charis/anaconda3/etc/profile.d/conda.sh
conda activate mai
# train
python utils/stage2/train_sgan.py  --cfg_file 'config/sgan_vgg16_321x321.yml'

# # inference with loader
# CUDA_VISIBLE_DEVICES=$1 python tools/stage2/infer_cam.py --cfg_file $2

# evaluate seed mIoU
# python tools/eval_mIoU.py --cfg_file $2