#!/bin/bash
cd /esat/izar/r0833114/SC-CAM
source /esat/christianso/r0833114/miniconda3/etc/profile.d/conda.sh
conda activate wsss
## 1) [Required] Please specify the path to the PSCAL VOC 2012 dataset (e.g., ~/datasets/VOCdevkit/VOC2012/)
dataset_folder=/esat/izar/r0833114/VOCdevkit/VOC2012
## 2) [Required] Please specify the path to the folder to save the feature/label/weight (e.g., ./save)
save_folder=results_voc
## 3) Please specify the path to the pretrained weight (It is default to save in the folder named weights)
pretrained_model=./pretrained_model/ilsvrc-cls_rna-a1_cls1000_ep-0001.params

## 4) Please specify the path to the model for each round
final_model=${save_folder}/weight/k10_R3_resnet_cls_ep50.pth \

## 5) Please specify the path to save the generated response map
save_cam_folder=${save_folder}/final_result

## Infer the classifier and generate the response map with the final model
python ./scripts/infer_cls.py --infer_list voc12/train_aug.txt --voc12_root ${dataset_folder} --weights ${final_model} --save_crf 1 --save_path ${save_cam_folder} --save_out_cam 1 --k_cluster 10 --round_nb 3
