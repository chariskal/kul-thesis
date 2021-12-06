import imageio
import sys
# sys.path.append('/content/drive/MyDrive/MAI/thesis/source/OAA')
sys.path.append('/home/charis/kul-thesis/OAA')
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.LoadData import test_data_loader
from utils.Restore import restore
import matplotlib.pyplot as plt
from models import vgg
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import utils.imutils as imutils

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    
def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--data_root", default="/esat/izar/r0833114/VOCdevkit/VOC2012", type=str)       # default is VOC2012
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--infer_list", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='voc2012')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--out_cam", default="./results_voc/results_cam", type=str)
    parser.add_argument("--out_crf", default="./results_voc/results_crf", type=str) 
    parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)

    return parser.parse_args()

def _crf_with_alpha(orig_img, cam_dict, alpha):
    v = np.array(list(cam_dict.values()))       # change dict to array
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]

    return n_crf_al

def get_model(args):
    model = vgg.vgg16(num_classes=args.num_classes)
    # model = torch.nn.DataParallel(model).cuda()

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()
    
    print(model_dict.keys())
    print(pretrained_dict.keys())
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return  model

def validate(args):
    print('\nvalidating ... ', flush=True, end='')
    crf_alpha = [4,24]

    model = get_model(args)
    model.eval()
    val_loader = test_data_loader(args)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, img, label_in = dat
            # print(dat)
            label = label_in.cuda(non_blocking=True)
            logits = model(img)
            last_featmaps = model.get_heatmaps()

            cv_im = cv2.imread(img_name[0])
            cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
            orig_img = np.asarray(Image.open(img_name[0]))                         # load img
            # orig_img_size = orig_img.shape[:2] 
            height, width = cv_im.shape[:2]
            #print(height, width)

            for l, featmap in enumerate(last_featmaps):
                cam_dict = {}
                maps = featmap.cpu().data.numpy()
                # print(maps)
                img_name_path = img_name[0].split('/')[-1][:-4]
                im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
                # print(im_name)
                labels = label_in.long().numpy()[0]
                for i in range(int(args.num_classes)):
                    if labels[i] == 1:
                        att = maps[i]
                        #print(att.shape)
                        att[att < 0] = 0
                        #print(att.shape)
                        cam_max = np.max(att, keepdims=True) # find min and max
                        cam_min = np.min(att, keepdims=True)
                        #print('max and min of cam: ',np.shape(cam_max), np.shape(cam_min))
                        att = att / (np.max(att) + 1e-8)

                        norm_cam = (att-cam_min-1e-5) / (cam_max - cam_min + 1e-5)
                        norm_cam = cv2.resize(norm_cam, (width, height), interpolation = cv2.INTER_CUBIC)
                        cam_dict[i] = norm_cam
                        att = np.array(att * 255, dtype=np.uint8)
                        out_name = im_name + '_{}.png'.format(i)
                        att = cv2.resize(att, (width, height), interpolation=cv2.INTER_CUBIC)
                        att = cv_im_gray * 0.2 + att * 0.8
                        #cv2.imwrite(out_name, att)
                        #print(out_name)
                        #plt.imsave(out_name, att, cmap=colormap(i))

                if args.out_cam is not None:
                    if not os.path.exists(args.out_cam):
                        os.makedirs(args.out_cam)
                    np.save(os.path.join(args.out_cam, img_name_path + '.npy'), cam_dict)        # save dictionary at ./results_cam
                    #print(os.path.join(args.out_cam, img_name_path + '.npy'))

                # if args.out_cam_pred is not None:
                #     if not os.path.exists(args.out_cam_pred):
                #         os.makedirs(args.out_cam_pred)
                #     bg_score = [np.ones_like(norm_cam[0])*args.out_cam_pred_alpha]          # make bg image with ones at corrct dimensions
                #     pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
                #     imageio.imwrite(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))

                if args.out_crf is not None:
                    if not os.path.exists(args.out_crf):
                        os.makedirs(args.out_crf)
                    for t in crf_alpha:
                        crf = _crf_with_alpha(orig_img, cam_dict, t)
                        folder = args.out_crf + ('_%.1f'%t)
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        np.save(os.path.join(folder, img_name_path + '.npy'), crf)


if __name__ == '__main__':
    args = get_arguments()
    validate(args)
