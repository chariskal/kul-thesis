# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
import PIL
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
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnest101', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)

parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration

    pred_dir = create_directory('./experiments/predictions/{}/'.format(args.tag))
    
    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'] + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(images, image_size):
        images = images.cuda()
        
        logits = model(images)
        logits = resize_for_tensors(logits, image_size)
        
        logits = logits[0] + logits[1].flip(-1)
        logits = get_numpy_from_tensor(logits).transpose((1, 2, 0))
        return logits

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                image = normalize_fn(image)
                image = image.transpose((2, 0, 1))

                image = torch.from_numpy(image)
                flipped_image = image.flip(-1)
                
                images = torch.stack([image, flipped_image])

                cams = inference(images, (ori_h, ori_w))
                cams_list.append(cams)
            
            preds = np.sum(cams_list, axis=0)
            preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                preds = crf_inference(np.asarray(ori_image), preds.transpose((2, 0, 1)), t=args.iteration)
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)

            ###############################################################################
            # cv2.imwrite('./demo.jpg', np.concatenate([np.asarray(ori_image)[..., ::-1], decode_from_colormap(pred_mask, dataset.colors)], axis=1))
            # input('write')

            # cv2.imshow('Image', np.asarray(ori_image)[..., ::-1])
            # cv2.imshow('Prediction', decode_from_colormap(pred_mask, dataset.colors))
            # cv2.imshow('GT', decode_from_colormap(gt_mask, dataset.colors))
            # cv2.waitKey(0)
            ###############################################################################

            if args.domain == 'test':
                pred_mask = decode_from_colormap(pred_mask, dataset.colors)[..., ::-1]
            
            imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))