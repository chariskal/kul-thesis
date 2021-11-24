# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import argparse
import PIL
import numpy as np
from models import vgg            # see ./models/vgg.py for backbone
import torch.optim as optim       # re-imports torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from traitlets.traitlets import default


from core.networks import Classifier
from core.datasets import VOC_Dataset_For_Making_CAM

from tools.general.io_utils import create_directory
from tools.general.time_utils import Timer
from tools.general.json_utils import read_json

from tools.ai.demo_utils import get_strided_size, get_strided_up_size
from tools.ai.torch_utils import set_seed, resize_for_tensors, calculate_parameters, load_model, load_ckp
from tools.ai.optim_utils import PolyOptimizer
from tools.ai.augment_utils import Normalize

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/home/charis/kul-thesis/VOCdevkit/VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--model_path', type=str, default='./voc12_epoch_13_acc8.pth')


def get_model(args):    # get modified VGG
    model = vgg.vgg16(pretrained=True, num_classes=20, att_dir='./', training_epoch=40)
    model = torch.nn.DataParallel(model).cuda()         # parallel GPU training
    param_groups = model.module.get_parameter_groups()
    optimizer = None
    # optimizer = optim.SGD([                             # use standard SGD
    #     {'params': param_groups[0], 'lr': args.lr},
    #     {'params': param_groups[1], 'lr': 2*args.lr},
    #     {'params': param_groups[2], 'lr': 10*args.lr},
    #     {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    return  model, optimizer

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s'%args.scales
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    # model_path = './experiments/models/' + f'{args.tag}.pth'

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
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(args.architecture, meta_dic['classes'], mode=args.mode)
    # param_groups = model.get_parameter_groups(print_fn=None)
    model, optimizer = get_model(args)

    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    # optimizer = PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
    #     {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
    #     {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    # ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=1, nesterov=args.nesterov)
    

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    # load_model(model, model_path, parallel=the_number_of_gpu > 1)
    model, _, current_epoch, global_counter = load_ckp(args.model_path, model, None)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def get_cam(ori_image, scale):
        # preprocessing
        image = copy.deepcopy(ori_image)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        images = images.cuda()
        
        # inferenece
        _, features = model(images, with_cam=True)

        # postprocessing
        cams = F.relu(features)
        cams = cams[0] + cams[1].flip(-1)

        return cams

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            npy_path = pred_dir + image_id + '.npy'
            if os.path.isfile(npy_path):
                continue
            
            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

            cams_list = [get_cam(ori_image, scale) for scale in scales]

            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)
            
            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]
            
            keys = torch.nonzero(torch.from_numpy(label))[:, 0]
            
            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            
            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

            # save cams
            keys = np.pad(keys + 1, (1, 0), mode='constant')
            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()))
            sys.stdout.flush()
        print()
    
    if args.domain == 'train_aug':
        args.domain = 'train'
    
    print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))