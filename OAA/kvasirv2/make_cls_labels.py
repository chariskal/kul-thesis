import argparse
import sys
sys.path.append('/content/drive/MyDrive/MAI/thesis/source/OAA')

import os
import data
import numpy as np
import random

def store_txt_files(args):
    ROOT_DIR = args.kvasir_root                         # kvasir root
    CLASSES_LIST = os.listdir(ROOT_DIR)                 # different class img in diff directories, name is class
    CLASSES_LIST.sort()
    ANNOT_FOLDER_NAME = "polyps/masks/"                 # only 'polyps' has seg annotations

    val_list = list(range(0,len(CLASSES_LIST)))       # value list for translating Class name to number
    cat_dict = dict(zip(CLASSES_LIST,val_list))         # create dictionary for translation

    f_train = open("train.txt","w+")                    # open files for storing the lists
    f_val = open("val.txt","w+")
    f_all = open("train_val.txt", "w+")
    f_train_seg = open("train_seg.txt", "w+")
    f_val_seg = open("val_seg.txt", "w+")
    f_all_seg = open("train_val_seg.txt", "w+")

    nline = '\n'
    list_of_paths = [ROOT_DIR+i for i in CLASSES_LIST]  # calculate the different paths for each class
    # Store all images and corresponding class in a list of tuples ('img_name',class_num) 
    kvasir_img_list = [(f,cat_dict[p[len(ROOT_DIR):]]) for p in list_of_paths for f in os.listdir(p) if f.endswith(".jpg")]

    num_images = len(kvasir_img_list)                   # find total number of img instances
    random.shuffle(kvasir_img_list)                     # randomly shuffle the list

    num1 = int(0.9*num_images)
    num2 = num_images - num1
    kvasir_train = kvasir_img_list[0:num1]
    kvasir_val = kvasir_img_list[num1+1:num_images]

    for img_name,cat in kvasir_img_list:
        folder_name = list(cat_dict.keys())[list(cat_dict.values()).index(cat)]
        if cat is 6:    # if polyps the also write for seg
            f_all_seg.write(f'{folder_name}/{img_name} {ANNOT_FOLDER_NAME}{img_name}\n')
        f_all.write(f'{folder_name}/{img_name} {cat}\n')

    for img_name,cat in kvasir_train:
        folder_name = list(cat_dict.keys())[list(cat_dict.values()).index(cat)]
        if cat is 6:    # if polyps the also write for seg
            f_train_seg.write(f'{folder_name}/{img_name} {ANNOT_FOLDER_NAME}{img_name}\n')
        f_train.write(f'{folder_name}/{img_name} {cat}\n')

    for img_name,cat in kvasir_val:
        folder_name = list(cat_dict.keys())[list(cat_dict.values()).index(cat)]
        if cat is 6:    # if polyps the also write for seg
            f_val_seg.write(f'{folder_name}/{img_name} {ANNOT_FOLDER_NAME}{img_name}\n')
        f_val.write(f'{folder_name}/{img_name} {cat}\n')

    f_all.close()
    f_val.close()
    f_train.close()
    f_val_seg.close()
    f_train_seg.close()
    f_all_seg.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=True, type=bool)
    parser.add_argument("--train_list", default='train.txt', type=str)
    parser.add_argument("--val_list", default='val.txt', type=str)
    parser.add_argument("--out", default="cls_labels.npy", type=str)
    parser.add_argument("--kvasir_root", default="/content/drive/MyDrive/MAI/thesis/source/kvasir-dataset-v2/", type=str)
    args = parser.parse_args()

    if args.store:
        store_txt_files(args)
    img_name_list,_ = data.load_img_name_list(args.train_list)
    tmp,_ = data.load_img_name_list(args.val_list)
    img_name_list = img_name_list + tmp
    print(type(img_name_list), len(img_name_list))

    label_list = data.load_label_list(args.train_list)
    label_list = label_list + data.load_label_list(args.val_list)
    print(type(label_list), len(label_list))

    d = dict()
    i = 0
    for img_name, label in zip(img_name_list, label_list):
        i = i+1
        #print(i, len(img_name), label)
        d[img_name] = label
 
    np.save(args.out, d)