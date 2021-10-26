#!/bin/sh
cd /esat/izar/r0833114/SGAN
source /esat/izar/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss
python utils/stage2/infer_cam.py --cfg_file 'config/sgan_vgg16_321x321.yml'