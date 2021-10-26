import numpy as np
import time
import os
from PIL import Image
import cv2
import logging
from os.path import exists 
import matplotlib as mpl
import matplotlib.pyplot as plt

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
threshold = 1.0

def colormap(index):
    # Return custom colormap
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    
def load_dataset(test_list):
    #logging.info('Beginning loading dataset...')
    img_list = []
    label_list = []
    with open(test_list) as f:
        test_names = f.readlines()
    lines = open(test_list).read().splitlines()
    for line in lines:
        fields = line.split()
        img_name = fields[0]
        img_labels = []
        for i in range(len(fields)-1):
            img_labels.append(int(fields[i+1]))
        img_list.append(img_name)
        label_list.append(img_labels)
    return img_list, label_list

def main():
    #print('Started ...')
    alpha = 0.20
    beta = 1-alpha
    dir1 = '/esat/izar/r0833114/SEAM/results_cam/'
    dir2 = '/esat/izar/r0833114/VOCdevkit/VOC2012/'
    dir3 = '/esat/izar/r0833114/SEAM/results_seg/'

    att_dir = '/esat/izar/r0833114/SEAM/results_cam/attentions'
    seg_dir = '/esat/izar/r0833114/SEAM/results_cam/seg'
    #train_lst = '/esat/izar/r0833114/VOCdevkit/VOC2012/ImageSets/Segmentation/val_oaa.txt'
    #img_list, label_list = load_dataset(train_lst)

    #os.chdir(dir1)
    for filename in os.listdir(dir1):
        if filename.endswith(".npy"):
            size = np.size(filename)
            filename2 = filename[:size - 5]
            #print(filename)
            name = '{}JPEGImages/{}.jpg'.format(dir2, filename2)
            img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)    #get original

            #print(name)
            cam_dict = np.load(dir1+filename, allow_pickle=True).item()
            height, width = list(cam_dict.values())[0].shape
            label_list = []
            #cam_tensor = np.zeros((21,height,width),np.float32)
            for label in cam_dict.keys():
                cam = cam_dict[label]
                label_list.append(label)
                att = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)

                min_value = np.min(att)
                max_value = np.max(att)
                att = (att - min_value) / (max_value - min_value + 1e-8)
                att = np.array(att*255, dtype = np.uint8)

                final = img*0.2 + att*0.8     
                save_name = '{}/{}_{}.png'.format(att_dir, filename2,label)
                #save_name2 = '{}/{}.png'.format(seg_dir, filename2)
                #print(save_name2)

                plt.imsave(save_name, final, cmap=colormap(label))
                #plt.imsave(save_name2, final, cmap=cv2.COLORMAP_HSV)
                break


if __name__ == '__main__':
        main()

                
        

