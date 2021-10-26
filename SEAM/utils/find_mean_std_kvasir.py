import numpy as np
import cv2
import random
import os
from tqdm import tqdm

# calculate means and std
train_txt_path = '/esat/izar/r0833114/OAA/kvasirv2/train_val.txt'
dataset_root = '/esat/izar/r0833114/kvasir_v2'

def get_img_path(img_name, dataset_root):
    return os.path.join(dataset_root, img_name)

def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-40:-4] for img_gt_name in img_gt_name_list]
    folder_paths_list = [img_gt_name.split(' ')[0] for img_gt_name in img_gt_name_list]
    return img_name_list, folder_paths_list


CNum = 7999 # How many images to choose for calculation
 
img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
 
with open(train_txt_path, 'r') as f:
    lines = f.readlines()
 
    for i in tqdm(range(CNum)):
        img_path, folder_path = load_img_name_list(train_txt_path)
 
        img = cv2.imread(get_img_path(folder_path[i], dataset_root))
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        
        imgs = np.concatenate((imgs, img), axis=3)
#         print(i)
 
    imgs = imgs.astype(np.float32)/255.
 
 
    for i in range(3):
         Pixels = imgs[:,:,i,:].ravel() # is drawn into a line
    means.append(np.mean(Pixels))
    stdevs.append(np.std(Pixels))
 
 
means.reverse() # BGR --> RGB
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))