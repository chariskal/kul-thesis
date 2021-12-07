import sys
sys.path.append('/home/charis/kul-thesis/SEAM')
import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import voc12.data
import kvasirv2.data
from utils import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo        # Loads the Torch serialized object at the given URL
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os
from torchvision import datasets, models

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        names = self.dataset[index][2]
        return x, y, names

    def __len__(self):
        return len(self.dataset)

    def set_transform(self, transform):
        self.transform = transform


def save_checkpoint(state, filename='checkpoint.pth.tar'):     # save points during training
    """Function for saving checkpoints"""
    snapshot_dir = './'
    savepath = os.path.join(snapshot_dir, filename)
    print(savepath)
    torch.save(state, savepath)
    print(f"Model saved to {savepath}")
    
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']+1, checkpoint['global_counter']

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def validate(model, criterion, total_samples):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()
  corrects = 0
  total_sum = 0
  total_loss = 0.0
  with torch.no_grad():
      for idx, dat in enumerate(test_data_loader,0):
          imgs, lbls, _ = dat
          imgs = imgs.to(device)
          lbls = lbls.to(device)

          outputs = model(imgs)
          val_loss = criterion(outputs, lbls)

          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total_sum += lbls.size(0)
          corrects += (predicted == lbls).sum().item()
          total_loss += loss.item()

  print('Acc val set: %f %%' % (
      100 * corrects / total_sum))
  print('Val loss: ', float(val_loss.cpu().numpy()))
  average_loss = total_loss / total_samples
  neptune.log_metric('val_loss', average_loss)
  return average_loss

if __name__ == '__main__':
    import neptune
    NEPTUNE_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMGUxMmQ1NC00ZDU4LTQ4ZGYtOWJjOC0xYTJkYjJmYmJiZDMifQ=='
    run = neptune.init(project_qualified_name='ch.kalavritinos/SEAM', api_token=NEPTUNE_TOKEN)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)                # batch size
    parser.add_argument("--max_epoches", default=8, type=int)               # maximum # of epochs   
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)     # use the default resnet38
    parser.add_argument("--lr", default=0.01, type=float)                   #learning rate
    parser.add_argument("--num_workers", default=12, type=int)               # number of workers    
    parser.add_argument("--wt_dec", default=5e-4, type=float)               # weight decay
    parser.add_argument("--train_list", default="kvasirv2/train.txt", type=str)    # list of training set
    parser.add_argument("--val_list", default="kvasirv2/val.txt", type=str)            # list of validation set
    parser.add_argument("--session_name", default="resnet38_SEAM", type=str)        # give this training sess a name
    parser.add_argument("--crop_size", default=448, type=int)                       # fixed crop size for the images
    parser.add_argument("--weights", required=True, type=str)                       # givethem the pre-trained weights
    parser.add_argument("--data_root", default='/home/charis/kul-thesis/kvasir-dataset-v2-new/', type=str)                # root of the VOC dir
    parser.add_argument("--tblog_dir", default='./tblog', type=str)                 # tblog dir    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pyutils.Logger(args.session_name + '.log')          # save a new log file
    print(vars(args))                                   # print input args
    
    model = getattr(importlib.import_module(args.network), 'Net')() # init model

    #print(model)                                        # print model args
    tblogger = SummaryWriter(args.tblog_dir)	        # print summary
    train_dir = args.data_root + "train/"
    test_dir = args.data_root + "test/"

    train_dataset = ImageFolderWithPaths(train_dir)
    val_dataset = ImageFolderWithPaths(test_dir)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    transform = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    }

    train_set = MyLazyDataset(train_dataset, transform["train"])
    val_set = MyLazyDataset(val_dataset, transform["test"])

    train_data_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_data_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )


    # mapping, train_data_loader = get_data_loader(data_dir=train_dir, batch_size=8, train=True)
    # mapping, test_data_loader = get_data_loader(data_dir=test_dir, batch_size=8, train=False)

    
    max_step = len(train_data_loader) * args.max_epoches             # calc step
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([              # use polyoptimizer 
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':              # if file ends with .params extension import resnet38d
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)           # load state dict
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    global_counter = 0
    best_val_loss = float('inf')
    current_epoch = 0

    print('Training started ...')
    PARAMS = {'dataset':'kvasir',
                'network':'vgg16',
                'epoch_nr': args.max_epoches,
                'batch_size': args.batch_size,
                'optimizer': 'SGD',
                'lr': args.lr
      }
    neptune.create_experiment('kvasirv2_train', params=PARAMS)

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')         # final loss = l_cls + l_er + l_ecr

    timer = pyutils.Timer("Session started: ")
    while current_epoch < args.max_epoches:
        total_loss = 0.0
        corrects = 0
        total_samples = 0
        for iter, dat in enumerate(train_data_loader):
            scale_factor = 0.3
            inputs, labels, fname = dat
            inputs = inputs.to(device)
            labels = labels.to(device)

            img1 = dat[0]
            img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=True) 
            N,C,H,W = img1.size()       # get dims
            label = labels             # get label
            #print(label)
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            #print(label)
            cam1, cam_rv1 = model(img1)
            #print(type(cam1), type(cam_rv1))
            #print(cam_rv1)
            label1 = F.adaptive_avg_pool2d(cam1, (1,1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
            cam1 = F.interpolate(visualization.max_norm(cam1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label

            cam2, cam_rv2 = model(img2)
            _, predicted = torch.max(cam2, 1)
            corrects += torch.sum(predicted == labels)
            total_samples += labels.size(0)
            label2 = F.adaptive_avg_pool2d(cam2, (1,1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
            cam2 = visualization.max_norm(cam2)*label
            cam_rv2 = visualization.max_norm(cam_rv2)*label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])

            ns,cs,hs,ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
            #loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
            cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
            cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]
#            with torch.no_grad():
#                eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)#*eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)#*eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(2*hs*ws*0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(2*hs*ws*0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2 
            loss = loss_cls + loss_er + loss_ecr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_counter += 1

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item()})

            if (optimizer.global_step - 1) % 50 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                # Visualization for training process
                img_8 = img1[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2,0,1))
                h = H//4; w = W//4
                p1 = F.interpolate(cam1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p2 = F.interpolate(cam2,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p_rv1 = F.interpolate(cam_rv1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p_rv2 = F.interpolate(cam_rv2,(h,w),mode='bilinear')[0].detach().cpu().numpy()

                image = cv2.resize(img_8, (w,h), interpolation=cv2.INTER_CUBIC).transpose((2,0,1))
                CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                #MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                loss_dict = {'loss':loss.item(), 
                             'loss_cls':loss_cls.item(),
                             'loss_er':loss_er.item(),
                             'loss_ecr':loss_ecr.item()}
                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                tblogger.add_image('Image', input_img, itr)
                #tblogger.add_image('Mask', MASK, itr)
                tblogger.add_image('CLS1', CLS1, itr)
                tblogger.add_image('CLS2', CLS2, itr)
                tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                tblogger.add_images('CAM1', CAM1, itr)
                tblogger.add_images('CAM2', CAM2, itr)
                tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                tblogger.add_images('CAM_RV2', CAM_RV2, itr)
        save_checkpoint(
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            },
                            filename='%s_epoch_%d.pth' %('kvasir_', current_epoch))
            
        current_epoch += 1

    # torch.save(model.module.state_dict(),'pretrained_model/' + args.session_name + '.pth')           # save final model

         
