import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from utils import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_data_loader(data_dir, batch_size=32, train=True):
    # define how we augment the data for composing the batch-dataset in train and test step
    transform = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    # ImageFolder with root directory and defined transformation methods for batch as well as data augmentation
    if train:
      data = ImageFolderWithPaths(root=data_dir, transform=transform['train'])
    else:
      data = ImageFolderWithPaths(root=data_dir, transform=transform['test'])
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=2)

    return data.class_to_idx, data_loader 

def make_folder(save_folder_path):
    if os.path.exists(save_folder_path) == False:
        os.mkdir(save_folder_path)
    if os.path.exists(os.path.join(save_folder_path, 'feature')) == False:
        os.mkdir(os.path.join(save_folder_path, 'feature'))
    if os.path.exists(os.path.join(save_folder_path, 'label')) == False:
        os.mkdir(os.path.join(save_folder_path, 'label'))
    if os.path.exists(os.path.join(save_folder_path, 'log')) == False:
        os.mkdir(os.path.join(save_folder_path, 'log'))
    if os.path.exists(os.path.join(save_folder_path, 'weight')) == False:
        os.mkdir(os.path.join(save_folder_path, 'weight'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, default="./weights/res38_cls.pth", type=str, help="the weight of the model")
    parser.add_argument("--network", default="network.resnet38_cls", type=str, help="the network of the classifier")
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str, help="the filename list for feature extraction")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--data_root", default="/content/drive/MyDrive/MAI/thesis/source/kvasir-dataset-v2", type=str, help="the path to the dataset folder")
    parser.add_argument("--from_round_nb", required=True, default=None, type=int, help="the round number of the extracter, e.g., 1st round: from_round_nb=0, 2nd round: from_round_nb=1, and so on")
    parser.add_argument("--k_cluster", default=10, type=int, help="the number of the sub-category")
    parser.add_argument("--save_folder", required=True, default='./results_kvasir', type=str, help="the path to save the extracted feature")
    parser.add_argument("--test_dir", default='/content/drive/My Drive/MAI/thesis/source/kvasir-dataset-v2-folds/1/test', type=str, help="the path to images")

    args = parser.parse_args()

    make_folder(args.save_folder)

    model = getattr(importlib.import_module(args.network), 'Net')(args.k_cluster, args.from_round_nb)
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    # infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, data_root=args.data_root,
    #                                                scales=(1, 0.5, 1.5, 2.0),
    #                                                inter_transform=torchvision.transforms.Compose(
    #                                                    [np.asarray,
    #                                                     model.normalize,
    #                                                     imutils.HWC_to_CHW
    #                                                     ]))

    # infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    torch.multiprocessing.freeze_support()

    # test_dir = '/content/drive/My Drive/MAI/thesis/source/kvasir-dataset-v2-folds/1/test'
    # test_dir2 = '/content/drive/My Drive/MAI/thesis/source/kvasir-dataset-v2-folds/1/train'

    mapp, infer_data_loader = get_data_loader(data_dir=args.test_dir, batch_size=args.batch_size, train=False)
    # mapp, test_data_loader2 = get_data_loader(data_dir=test_dir2, batch_size=args.batch_size, train=False)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    filename_list = []
    image_feature_list = []

    print('################################ Extracting features from Round-{} ...... ################################'.format(args.from_round_nb))

    for iter, (imgs, lbls, fname) in enumerate(infer_data_loader, 0):
        # extract feature
        if args.from_round_nb == 0:
            tmp, feature, _ = model.forward(imgs.cuda(), args.from_round_nb)
        else:
            tmp, feature, y_8, x_80, y_80 = model.forward(imgs.cuda(), args.from_round_nb)

        feature = feature[0].cpu().detach().numpy()
        image_feature_list.append(feature)

        if iter % 500 == 0:
            print('Already extracted: {}/{}'.format(iter, len(infer_data_loader)))


    image_feature_list = np.array(image_feature_list)
    print(image_feature_list.shape)

    # save the extracted feature
    save_feature_folder_path = os.path.join(args.save_folder, 'feature')
    feature_save_path = os.path.join(save_feature_folder_path, 'R{}_feature.npy'.format(args.from_round_nb)) # R1 feature is for R2 use
    np.save(feature_save_path, image_feature_list)
    print('extract_feature.py done')
