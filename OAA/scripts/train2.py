import sys
sys.path.append('/home/charis/kul-thesis/OAA')
import torch
import numpy as np
import argparse                   # argument parser
import os
import time
import shutil                     # high level file operation
import json
import my_optim                   # see ./scripts/my_optim.py
import torch.optim as optim       # re-imports torch.optim as optim
from models import vgg            # see ./models/vgg.py for backbone
import torch.nn as nn     
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter    # see ./utils/avg.Meter.py
from utils.LoadData import train_data_loader      # see ./utils/LoadData.py
from tqdm import trange, tqdm
import kvasirv2.data
import utils.imutils as imutils
import neptune

NEPTUNE_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMGUxMmQ1NC00ZDU4LTQ4ZGYtOWJjOC0xYTJkYjJmYmJiZDMifQ=='


ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)


def worker_init_fn(worker_id):
    np.random.seed(1 + worker_id)


# Get arguments defined in bash script
def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of OAA')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--data_root", type=str, default='/esat/izar/r0833114/kvasir_v2/', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='kvasirv2/train.txt')     # list of train imgs
    parser.add_argument("--test_list", type=str, default='kvasirv2/val.txt')      # list of test imgs
    parser.add_argument("--batch_size", type=int, default=20)         # batch size
    parser.add_argument("--input_size", type=int, default=256)        # image input size
    parser.add_argument("--crop_size", type=int, default=224)         # crop images
    parser.add_argument("--dataset", type=str, default='kvasir')      # default imagenet but using pascal voc
    parser.add_argument("--num_classes", type=int, default=20)        # choose how many classes
    parser.add_argument("--threshold", type=float, default=0.6)       # what is this threshold?
    parser.add_argument("--lr", type=float, default=0.001)            # learning rate
    parser.add_argument("--weight_decay", type=float, default=0.0005) # weight decay factor
    parser.add_argument("--decay_points", type=str, default='61')     # numberof decay points
    parser.add_argument("--epoch", type=int, default=15)              # number of epochs
    parser.add_argument("--num_workers", type=int, default=20)        # number of workers
    parser.add_argument("--disp_interval", type=int, default=100)     # display interval
    parser.add_argument("--snapshot_dir", type=str, default='checkpoints/train/')       # where to store snapshots
    parser.add_argument("--resume", type=str, default='False')        # resume training 
    parser.add_argument("--global_counter", type=int, default=0)      
    parser.add_argument("--current_epoch", type=int, default=0)       # number of current epoch upon resuming training i guess
    parser.add_argument("--att_dir", type=str, default='./results_voc/')     # directory attributes are stored

    return parser.parse_args()    # parse argu

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):     #save points during training
    savepath = os.path.join(args.snapshot_dir, filename)
    print(savepath)
    torch.save(state, savepath)
    if is_best:     #if current epoch is best, then save checkpoint
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):    # get modified VGG
    model = vgg.vgg16(pretrained=True, num_classes=args.num_classes, att_dir=args.att_dir, training_epoch=args.epoch)
    model = torch.nn.DataParallel(model).cuda()         # parallel GPU training
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([                             # use standard SGD
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    return  model, optimizer

def validate(model, val_loader):
    print('\nvalidating ... ', flush=True, end='')
    val_loss = AverageMeter()
    model.eval()
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, img, label = dat
            label = label.cuda(non_blocking=True)
            logits = model(img)
            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)
            loss_val = F.multilabel_soft_margin_loss(logits, label)   
            val_loss.update(loss_val.data.item(), img.size()[0])

    print('validating loss:', val_loss.avg)

def train(args):      # start training function
    # parameters
    batch_time = AverageMeter()                             # Computes and stores the average and current value, can also resets to 0
    losses = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    model, optimizer = get_model(args)                      # call get_model(), get modified VGG and optimizer

    mean_vals = [0.485, 0.485, 0.485]                       # find actual mean values
    std_vals = [0.335, 0.335, 0.335]

    train_dataset = kvasirv2.data.KvasirClsDataset(args.train_list, dataset_root=args.data_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        transforms.Normalize(mean_vals, std_vals),
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))  # get class for train dataset

    test_dataset = kvasirv2.data.KvasirClsDataset(args.test_list, dataset_root=args.data_root,
                                transform=transforms.Compose([
        imutils.RandomResizeLong(448, 768),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        np.asarray,
        transforms.Normalize(mean_vals, std_vals),
        imutils.RandomCrop(args.crop_size),
        imutils.HWC_to_CHW,
        torch.from_numpy
    ]))  # get class for test dataset


    # train_loader, val_loader = train_data_loader(args)      # load data
    run = neptune.init(project_qualified_name='ch.kalavritinos/OAA', api_token=NEPTUNE_TOKEN)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    val_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    print(model)
    model.train()                                           # actually train the model
    end = time.time()                                       # check how much time it took

    # PARAMS = {'dataset':args.dataset,
    #                 'epoch_nr': args.max_epoches,
    #                 'batch_size': args.batch_size,
    #                 'optimizer': 'SGD',
    #                 'lr1': args.lr,}

    neptune.create_experiment('train_cam')#, params=PARAMS)


    while current_epoch < total_epoch:
        model.train()
        losses.reset()                                      # reset losses to 0
        batch_time.reset()    
        # res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)
        
        validate(model, val_loader)                         # loss validation
        index = 0  
        for idx, dat in enumerate(train_loader):
            img_name, img, label = dat
            label = label.cuda(non_blocking=True)
            #print(f'img shape: {np.shape(img)}')
            logits = model(img, current_epoch, label, index)
            index += args.batch_size

            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)
            loss_val = F.multilabel_soft_margin_loss(logits, label)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            losses.update(loss_val.data.item(), img.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))
                neptune.log_metric('val_loss', loss_val.data.item())
                neptune.log_metric('lr', optimizer.param_groups[0]['lr'])

        if current_epoch == args.epoch-1:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
