import sys
sys.path.append('/esat/izar/r0833114/SEAM')
import numpy as np
import torch
import cv2
import os
import voc12.data
import kvasirv2.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from utils import imutils, pyutils, visualization
import argparse
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import imageio

"""
Use pre-trained resnet38 to infer CAMs. Store resulting .npy arrays in separate directories
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="resnet38_exp1.pth", type=str)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--data_root", default="/esat/izar/r0833114/VOCdevkit/VOC2012", type=str)       # default is VOC2012
    parser.add_argument("--out_cam", default="./results_voc/results_cam", type=str)
    parser.add_argument("--out_crf", default="./results_voc/results_crf", type=str) 
    parser.add_argument("--out_cam_pred", default='./results_voc/results_cam_pred', type=str)
    parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)
    parser.add_argument("--num_classes", default=20, type=int)

    args = parser.parse_args()

    crf_alpha = [4,24]
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))
    num_classes = args.num_classes

    model.eval()
    model.cuda()
    # get custom KvasirDataset MSF with mutliple scales to check transformations
    infer_dataset = kvasirv2.data.KvasirClsDatasetMSF(args.infer_list, data_root=args.data_root,
                                                  scales=[0.5, 1.0, 1.5, 2.0],
                                                  inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW])) # changeformat from height x width x color
    _,f_list  = kvasirv2.data.load_img_name_list(args.infer_list)
    #print(type(f_list), len(f_list), f_list[0])
    # infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.data_root,
    #                                               scales=[0.5, 1.0, 1.5, 2.0],
    #                                               inter_transform=torchvision.transforms.Compose(
    #                                                    [np.asarray,
    #                                                     model.normalize,
    #                                                     imutils.HWC_to_CHW])) # changeformat from height x width x color
    
    #print("ready to load data ...")
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True) # load data
    #print("data loaded! ")
    n_gpus = torch.cuda.device_count()
    #print(f'n_gpus: {n_gpus} and type {type(n_gpus)}')
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    #print("starting iterations ...")
    i=0
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]
        # print('Label: ', label, 'img name:', img_name, args.data_root)

        img_path = kvasirv2.data.get_img_path(f_list[i], args.data_root)       # get path
        # print(img_path, img_name)
        i=i+1
        orig_img = np.asarray(Image.open(img_path))                         # load img
        orig_img_size = orig_img.shape[:2]                                  # get size

        def _work(i, img):                                                  # get cams for rescaled imgs
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    _, cam = model_replicas[i%n_gpus](img.cuda())
                    #print('cam_shape is : ', np.shape(cam))
                    cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                    #print('can upsampled is: ', np.shape(cam))
                    cam = cam.cpu().numpy() * label.clone().view(num_classes, 1, 1).numpy()
                    #print('final cam:', np.shape(cam))
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=1, prefetch_size=4, processes=args.num_workers)
        #print(np.unique(np.array(label)))
        cam_list = thread_pool.pop_results()            # get list of all cams
        #print('length of cam list: ', len(cam_list))
        sum_cam = np.sum(cam_list, axis=0)              #  find sum, if negative then bound it to 0
        #print('sum_cam shape:',np.shape(sum_cam))
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1,2), keepdims=True) # find min and max
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        #print('max and min of cam: ',np.shape(cam_max), np.shape(cam_min))
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)          # normalize cams
        #print('norm cam shape: ',np.shape(norm_cam), 'norm_cam max: ', np.max(norm_cam), 'and min: ', np.min(norm_cam))

        cam_dict = {}                       # init empty dictionary, key is class, value is the array
        for i in range(num_classes):                 # 1 labels (classes)
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]
        #print('cam_dict: ',cam_dict)
        if args.out_cam is not None:
            if not os.path.exists(args.out_cam):
                os.makedirs(args.out_cam)
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)        # save dictionary at ./results_cam

        if args.out_cam_pred is not None:
            if not os.path.exists(args.out_cam_pred):
                os.makedirs(args.out_cam_pred)
            bg_score = [np.ones_like(norm_cam[0])*args.out_cam_pred_alpha]          # make bg image with ones at corrct dimensions
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            imageio.imwrite(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))       # change dict to array
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_crf is not None:
            if not os.path.exists(args.out_crf):
                os.makedirs(args.out_crf)
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t)
                folder = args.out_crf + ('_%.1f'%t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)
