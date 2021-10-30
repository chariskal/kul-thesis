import os
import random

ROOT_DIR = '/esat/izar/r0833114/kvasir_v2/'         # kvasir root
CLASSES_LIST = os.listdir(ROOT_DIR)                 # different class img in diff directories, name is class
CLASSES_LIST.sort()
ANNOT_FOLDER_NAME = "polyps/masks/"                 # only 'polyps' has seg annotations

val_list = list(range(1,len(CLASSES_LIST)+1))       # value list for translating Class name to number
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
    if cat is 7:    # if polyps the also write for seg
        f_all_seg.write(f'{folder_name}/{img_name} {ANNOT_FOLDER_NAME}{img_name}\n')
    f_all.write(f'{folder_name}/{img_name} {cat}\n')

for img_name,cat in kvasir_train:
    folder_name = list(cat_dict.keys())[list(cat_dict.values()).index(cat)]
    if cat is 7:    # if polyps the also write for seg
        f_train_seg.write(f'{folder_name}/{img_name} {ANNOT_FOLDER_NAME}{img_name}\n')
    f_train.write(f'{folder_name}/{img_name} {cat}\n')

for img_name,cat in kvasir_val:
    folder_name = list(cat_dict.keys())[list(cat_dict.values()).index(cat)]
    if cat is 7:    # if polyps the also write for seg
        f_val_seg.write(f'{folder_name}/{img_name} {ANNOT_FOLDER_NAME}{img_name}\n')
    f_val.write(f'{folder_name}/{img_name} {cat}\n')

f_all.close()
f_val.close()
f_train.close()
f_val_seg.close()
f_train_seg.close()
f_all_seg.close()