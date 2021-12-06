import numpy as np
import time
import os
from PIL import Image
import cv2
import logging
from os.path import exists 
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io
from skimage import color

colormaps = ['#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']
# colormaps_rgb = [(127,0,0), (0,127,0), (127,127,0), (0,0,127), (127,0,127), (0,127,127), (127,127,127), (63,0,0), (191,0,0), (63,127,0), 
#         (191,127,0), (63,0,127), (191,0,127), (63,127,127), (191,127,127), (0,63,0), (127,63,0), (0,191,0), (127,191,0), (0,63,127)]

colormaps_rgb = [tuple(int(el.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for el in colormaps]

categories = ['background', 'aeroplane','bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def colormap(index):
    # Return custom colormap
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)


def main():
    #print('Started ...')
    # print(type(colormaps))
    alpha = 0.20
    beta = 1 - alpha
    
    dir2 = '/home/charis/kul-thesis/VOCdevkit/VOC2012/'
    dir3 = '/home/charis/kul-thesis/OAA/scripts/experiments/predictions/DeepLabv3+@ResNet-101@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10/'

    seg_dir = '/home/charis/kul-thesis/OAA/scripts/experiments/predictions/DeepLabv3+@ResNet-101@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10/seg'

    for filename in os.listdir(dir3):
        if filename.endswith(".png"):
            size = np.size(filename)
            filename2 = filename[:size - 5]
            # print(filename)
            name = '{}JPEGImages/{}.jpg'.format(dir2, filename2)
            img = cv2.imread(name)    #get original
            #print(name)
            seg = cv2.imread(dir3+filename, cv2.IMREAD_GRAYSCALE)
            #print(seg.shape, img.shape)
            height, width = seg.shape[0],seg.shape[1]
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            a = color.label2rgb(seg, img, colors=colormaps_rgb, alpha=0.3, bg_label=0, kind='avg')
            a = a*0.8 + img*0.2
            #print(type(a),a)
            save_name = '{}/{}.png'.format(seg_dir, filename2)
            # print(save_name)
            min_value = np.min(a)
            max_value = np.max(a)
            a = (a - min_value) / (max_value - min_value + 1e-8)
            a = np.array(a*255, dtype = np.uint8)
            cv2.imwrite(save_name, a)
            #plt.imsave(save_name2, final, cmap=cv2.COLORMAP_HSV)

if __name__ == '__main__':
        main()

                
        

