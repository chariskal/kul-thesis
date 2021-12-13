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
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from traitlets.traitlets import default
from torchvision import models, transforms


from core.networks import Classifier
from core.datasets import VOC_Dataset_For_Making_CAM
from core.datasets import Kvasir_Dataset, Kvasir_Dataset_For_Making_CAM

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
# parser.add_argument('--data_dir', default='/home/charis/kul-thesis/VOCdevkit/VOC2012/', type=str)
parser.add_argument('--data_dir', default='/home/charis/kul-thesis/kvasir-dataset-v2-new/', type=str)
parser.add_argument('--att_dir', default='./results_kvasir/exp10', type=str)
parser.add_argument('--epoch', default=20, type=int)


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
parser.add_argument('--model_path', type=str, default='/home/charis/kul-thesis/OAA/checkpoints/train/exp10/kvasir_epoch_39.pth')

import torch.nn as nn

class CustomResNet(nn.Module):
    def __init__(self, num_classes=20, init_weights=True, att_dir='./results_kvasir/exp10', training_epoch=15):
        super(CustomResNet, self).__init__()
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model = torch.nn.Sequential(*(list(self.resnet_model.children())[:-1]))
        self.extra_convs = nn.Sequential(
                                nn.Conv2d(self.resnet_num_ftrs, 256, kernel_size=3, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(256, num_classes, 1)           
                            )
        
        self.nclasses = num_classes
        self._initialize_weights()
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir)
        
    def forward(self, x, epoch=1, label=None, index=None):
        x = self.resnet_model(x)
        x = self.extra_convs(x)
        
        self.map1 = x.clone().detach()
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.nclasses)
        
        ###  the online attention accumulation process
        pre_probs = x.clone().detach()
        probs = torch.sigmoid(pre_probs)  # compute the prob
        pred_inds_sort = torch.argsort(-probs)

        if index != None and epoch > 0:
            atts = self.map1
            atts[atts < 0] = 0
            ind = torch.nonzero(label)
            num_labels = torch.sum(label, dim=1).long()

            for i in range(ind.shape[0]):
                batch_index, la = ind[i]
                pred_ind_select = pred_inds_sort[batch_index, :num_labels[batch_index]]

                accu_map_name = '{}/{}_{}.png'.format(self.att_dir, batch_index+index, la)
                att = atts[batch_index, la].cpu().data.numpy()
                att = att / (att.max() + 1e-8) * 255
                
                # if this is the last epoch and the image without any accumulation
                if epoch == self.training_epoch - 1 and not os.path.exists(accu_map_name):
                    cv2.imwrite(accu_map_name, att)
                    continue
                
                #naive filter out the low quality attention map with prob
                if la not in list(pred_ind_select):  
                    continue

                if not os.path.exists(accu_map_name):
                    cv2.imwrite(accu_map_name, att)
                else:
                    accu_att = cv2.imread(accu_map_name, 0)
                    accu_att = np.maximum(accu_att, att)
                    cv2.imwrite(accu_map_name,  accu_att)
         ##############################################

        return x

    def get_heatmaps(self):
        return self.map1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups

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

def prepare_model_custom(args):
    """Function for getting ResNet152"""
    model = CustomResNet(num_classes=8, att_dir=args.att_dir, training_epoch=args.epoch)
    model = torch.nn.DataParallel(model).cuda()                         # parallel GPU training
    param_groups = model.module.get_parameter_groups()
    lr = 0.001

    optimizer = optim.SGD([{'params': param_groups[0], 'lr': lr},
        {'params': param_groups[1], 'lr': 4*lr},
        {'params': param_groups[2], 'lr': 15*lr},
        {'params': param_groups[3], 'lr': 30*lr}]
        , lr=lr,  weight_decay=0.0005)
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
    # meta_dic = read_json('./data/VOC_2012.json')
    meta_dic = read_json('./data_kvasir/kvasir.json')
    # dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    dataset = Kvasir_Dataset_For_Making_CAM(args.data_dir, args.domain)
    ###################################################################################
    # Network
    ###################################################################################
    # model = Classifier(args.architecture, meta_dic['classes'], mode=args.mode)
    # param_groups = model.get_parameter_groups(print_fn=None)
    model, optimizer = prepare_model_custom(args)

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
        image = image.cuda()
        _, features = model(image)
        print('features:', features.shape)
        # cams1 = model.module.get_heatmaps() 

        # image = torch.from_numpy(image)
        # flipped_image = image.flip(-1)
        # print('cams.shape:', cams1.shape)
        # # images = torch.stack([image, flipped_image])
        
        
        # # inferenece
        # _, features = model(flipped_image)
        # print('features:', features.shape)
        # cams2 = model.module.get_heatmaps() 

        # # postprocessing
        cams = F.relu(features)
        # print('cams.shape:', cams2.shape)
        # cams = cams1 + cams2.flip(-1)

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