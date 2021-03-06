# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import argparse
import numpy as np
import cv2
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.ai.demo_utils import get_strided_size, get_strided_up_size
import PIL
from core.networks import AffinityNet
from core.datasets import VOC_Dataset_For_Making_CAM, Kvasir_Dataset_For_Making_CAM
from tools.general.io_utils import create_directory
from tools.general.time_utils import Timer
from tools.general.json_utils import read_json
from tools.ai.torch_utils import set_seed, resize_for_tensors, calculate_parameters, load_model, load_ckp

from tools.ai.torch_utils import set_seed, calculate_parameters, load_model
from core.aff_utils import PathIndex, propagate_to_edge

from tools.ai.augment_utils import Normalize


def image_resize(image, width = None, height = None, inter = cv2.INTER_CUBIC):
    dim = None
    (h, w) = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (np.ceil(w * r), height)

    else:
        r = width / float(w)
        # print(h, r, h*r, np.ceil(h*r))
        dim = (width, int(np.ceil(h * r)))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--data_dir', default='/home/charis/kul-thesis/kvasir-dataset-v2-new/', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--model_name', default='', type=str)
parser.add_argument('--cam_path', default='/home/charis/kul-thesis/OAA/results_kvasir/exp10/results_cam/', type=str)

parser.add_argument('--cam_dir', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--exp_times', default=8, type=int)
# parser.add_argument('--threshold', default=0.25, type=float)

def get_cam(cam, scale):
        # preprocessing
        ori_w, ori_h = cam.size
        image = copy.deepcopy(cam)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        images = images.cuda()
        
        # inference
        _, features = model(images, with_cam=True)
        print('features: ', features.shape)

        # postprocessing
        cams = F.relu(features)
        print('cams shape: ', cams.shape)
        cams = cams[0] + cams[1].flip(-1)
        print('cams shape: ', cams.shape)

        return cams

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    experiment_name = args.model_name

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@beta=%d'%args.beta
    experiment_name += '@exp_times=%d'%args.exp_times
    # experiment_name += '@threshold=%.2f'%args.threshold
    experiment_name += '@rw'
    
    # cam_dir = f'./experiments/predictions/{args.cam_dir}/'
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.model_name}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    # meta_dic = read_json('./data/VOC_2012.json')
    meta_dic = read_json('./data_kvasir/kvasir.json')
    dataset = Kvasir_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    path_index = PathIndex(radius=10, default_size=(512 // 4, 512 // 4))
    model = AffinityNet(args.architecture, path_index)

    model = model.cuda()
    model.eval()
    
    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            # preprocessing
            image = np.asarray(ori_image)
            image = normalize_fn(image)
            image = image.transpose((2, 0, 1))

            image = torch.from_numpy(image)
            flipped_image = image.flip(-1)

            images = torch.stack([image, flipped_image])
            images = images.cuda()

            # inference
            edge = model.get_edge(images)
            print(type(edge), edge.shape)

            npy_path = pred_dir + image_id + '.npy'
            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)
            print(strided_size, strided_up_size)


            # postprocessing
            cam_dict = np.load(args.cam_path + image_id.split('/')[-1] + '.npy', allow_pickle=True).item()
            # print(cam_dict[0].shape)
            keys_temp = list(cam_dict.keys())
            # print(keys_temp)
            keys = [0]
            for item in keys_temp:
                keys.append(item.cpu()+1)
                # print(keys)
            # keys = keys.cpu().numpy()
            keys = np.array(keys)

            t = []
            for i, key in enumerate(cam_dict.keys()):
                    # print(cv2.resize(cam_dict[key], (strided_size[1], strided_size[0]), interpolation = cv2.INTER_CUBIC).shape)
                    t.append(cv2.resize(cam_dict[key], (strided_size[1], strided_size[0]), interpolation = cv2.INTER_CUBIC))
                    # t.append(image_resize(cam_dict[key], width=125))
            cams = np.stack(t, axis=0)
            cams = torch.from_numpy(cams)
            print('cams shape: ', cams.shape)

            # cams = cam_dict['cam']
            
            cam_downsized_values = cams.cuda()
            print(cam_downsized_values.shape)
            rw = propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            print('rw: ', rw.shape)
            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :ori_h, :ori_w]
            rw_up = rw_up / torch.max(rw_up)
            
            np.save(npy_path, {"keys": cam_dict['keys'], "rw": rw_up.cpu().numpy()})
            
            sys.stdout.write('\r# Make CAM with Random Walk [{}/{}] = {:.2f}%, ({}, rw_up={}, rw={})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), rw_up.size(), rw.size()))
            sys.stdout.flush()
        print()
    
    print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))