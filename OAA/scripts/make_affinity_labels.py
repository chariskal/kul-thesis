# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from core.puzzle_utils import tile_features, merge_features, puzzle_module
from core.networks import FixedBatchNorm, Backbone, Classifier, Classifier_For_Positive_Pooling, Classifier_For_Puzzle, AffinityNet, Decoder, DeepLabv3_Plus, Seg_Model, SynchronizedBatchNorm2d, CSeg_Model
from core.datasets import VOC_Dataset, VOC_Dataset_For_Affinity, VOC_Dataset_For_Classification, VOC_Dataset_For_Evaluation, VOC_Dataset_For_Making_CAM, VOC_Dataset_For_Segmentation, VOC_Dataset_For_Testing_CAM, VOC_Dataset_For_WSSS

from tools.general.io_utils import create_directory, str2bool
from tools.general.time_utils import get_today, Timer
from tools.general.json_utils import read_json, write_json

from tools.ai.log_utils import Logger, Average_Meter
from tools.ai.demo_utils import get_strided_size, get_strided_up_size, imshow, transpose, denormalize, colormap, decode_from_colormap, normalize, crf_inference, crf_inference_label, crf_with_alpha
from tools.ai.optim_utils import PolyOptimizer
from tools.ai.torch_utils import set_seed, rotation, interleave, de_interleave, resize_for_tensors, L1_Loss, L2_Loss, Online_Hard_Example_Mining, shannon_entropy_loss, make_cam, one_hot_embedding, calculate_parameters, get_learning_rate, get_learning_rate_from_optimizer, get_cosine_schedule_with_warmup, get_numpy_from_tensor, load_model, save_model, transfer_model 
from tools.ai.evaluate_utils import calculate_for_tags, calculate_mIoU, Calculator_For_mIoU

from tools.ai.augment_utils import convert_OpenCV_to_PIL, convert_PIL_to_OpenCV, RandomCrop, RandomCrop_For_Segmentation, RandomHorizontalFlip, RandomHorizontalFlip_For_Segmentation, RandomResize, RandomResize_For_Segmentation, Resize_For_Mask, Normalize, Normalize_For_Segmentation, Top_Left_Crop, Top_Left_Crop_For_Segmentation, Transpose, Transpose_For_Segmentation
from tools.ai.randaugment import AutoContrast, RandAugmentMC, RandAugmentPC, Rotate, Color, Contrast, Cutout, CutoutAbs, Equalize, Identity, Invert, Posterize, Sharpness, ShearX, ShearY, Solarize, SolarizeAdd, TranslateX, TranslateY, _float_parameter, _int_parameter, fixmatch_augment_pool, my_augment_pool

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='/home/charis/kul-thesis/VOCdevkit/VOC2012/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--cam_path', default='/home/charis/kul-thesis/OAA/results_voc/exp2/results_cam/', type=str)
parser.add_argument('--domain', default='train_aug', type=str)

parser.add_argument('--fg_threshold', default=0.40, type=float)
parser.add_argument('--bg_threshold', default=0.10, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = 'ResNet50@Puzzle@acc@train@scale=0.5,1.0,1.5,2.0'
    
    pred_dir = f'./experiments/predictions/{experiment_name}/'
    aff_dir = create_directory('./experiments/predictions/{}@aff_fg={:.2f}_bg={:.2f}/'.format(experiment_name, args.fg_threshold, args.bg_threshold))

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    #################################################################################################
    # Convert
    #################################################################################################
    eval_timer = Timer()
    
    length = len(dataset)
    for step, (ori_image, image_id, _, _) in enumerate(dataset):
        png_path = aff_dir + image_id + '.png'
        if os.path.isfile(png_path):
            continue

        # load
        image = np.asarray(ori_image)
        cam_dict_arr = np.load(args.cam_path + image_id + '.npy', allow_pickle=True)
        cam_dict = cam_dict_arr[()]

        ori_h, ori_w, c = image.shape
        
        keys_temp = list(cam_dict.keys())
        keys = [0]
        for item in keys_temp:
            keys.append(item+1)
        keys = np.array(keys)
        # print(keys)

        t = []
        for i, key in enumerate(cam_dict.keys()):
                # print(cam_file[key].shape)
                t.append(cam_dict[key])
        cams = np.stack(t, axis=0)
        # print(cams.shape)
        # cams = cam_dict['hr_cam']

        # 1. find confident fg & bg
        fg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
        fg_cam = np.argmax(fg_cam, axis=0)
        fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]
        
        bg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bg_threshold)
        bg_cam = np.argmax(bg_cam, axis=0)
        bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]
        
        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0
        
        imageio.imwrite(png_path, conf.astype(np.uint8))
        
        sys.stdout.write('\r# Convert [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
        sys.stdout.flush()
    print()