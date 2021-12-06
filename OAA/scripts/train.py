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
import torch.nn.functional as F
from utils import AverageMeter    # see ./utils/avg.Meter.py
from utils.LoadData import train_data_loader      # see ./utils/LoadData.py
from tqdm import trange, tqdm
import kvasirv2.data
import utils.imutils as imutils
import neptune
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision

NEPTUNE_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMGUxMmQ1NC00ZDU4LTQ4ZGYtOWJjOC0xYTJkYjJmYmJiZDMifQ=='


ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

class ExLoss(nn.Module):
    def __init__(self):
        super(ExLoss, self).__init__()

    def forward(self, input, target):
        print(input.size(), target.size())
        assert(input.size() == target.size())
        pos = torch.gt(target, 0.001)
        neg = torch.le(target, 0.001)
        pos_loss = -target[pos] * torch.log(torch.sigmoid(input[pos]))
        neg_loss = -torch.log(1 - torch.sigmoid(input[neg]) + 1e-8)
      
        loss = 0.0
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        # print(num_pos, num_neg)
        if num_pos > 0:
            loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
        if num_neg > 0:
            loss += 1.0 / num_neg.float() * torch.sum(neg_loss)
      
        return loss

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
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

def save_checkpoint(args, state, filename='checkpoint.pth.tar'):     #save points during training
    savepath = os.path.join(args.snapshot_dir, filename)
    print(savepath)
    torch.save(state, savepath)

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
    
def validate(model, val_loader, loss, criterion, total_samples):
    model.eval()
    corrects = 0
    total_sum = 0
    total_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for idx, dat in enumerate(val_loader,0):
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

def train(args):      # start training function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # criterion = ExLoss()
    criterion = F.multilabel_soft_margin_loss
    # criterion = torch.nn.CrossEntropyLoss()

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    model, optimizer = get_model(args)                      # call get_model(), get modified VGG and optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)
    mapping, train_loader = get_data_loader(data_dir=args.data_root+'train', batch_size=args.batch_size, train=True)
    mapping, val_loader = get_data_loader(data_dir=args.data_root+'test', batch_size=args.batch_size, train=False)

    import matplotlib.pyplot as plt
    import numpy as np

    # function to show an imag
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    # dataiter = iter(val_loader)
    # images, labels, _ = dataiter.next()
    

    run = neptune.init(project_qualified_name='ch.kalavritinos/OAA', api_token=NEPTUNE_TOKEN)

    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    # print(model)
    model.train()                                           # actually train the model
    # end = time.time()                                       # check how much time it took

    global_counter = 0
    best_val_loss = float('inf')
    current_epoch = 0

    print('Training started ...')
    PARAMS = {'dataset':args.dataset,
                'network':'vgg16',
                'epoch_nr': args.epoch,
                'batch_size': args.batch_size,
                'optimizer': 'SGD',
                'lr': args.lr
        }
    neptune.create_experiment('kvasirv2_train', params=PARAMS)

    while current_epoch < total_epoch:
        model.train()
        # losses.reset()                                      # reset losses to 0
        # batch_time.reset()    
        total_loss = 0.0
        corrects = 0
        total_samples = 0
        
        # validate(model, val_loader)                         # loss validation
        index = 0  
        for idx, dat in enumerate(train_loader):
            img, label, fname = dat
            img = img.to(device)
            label = label.to(device)
            print(img.max(), img.min())
            # label = torch.reshape(label, (label.shape[0], 1))
            # print(img.shape, img.size())
            # print(label.shape, label.size())
            # imshow(torchvision.utils.make_grid(img.cpu()))
            optimizer.zero_grad()
            # forward + backward + optimize
            logits = model(img, current_epoch, label, index)
            index += args.batch_size
            print(logits.max(), logits.min())
            if len(logits.shape) == 1:
                print("YES")
                logits = logits.reshape(label.shape)
            # print(logits.shape)
            # print(logits == logits, label == label)

            loss = criterion(logits, label)
            _, predicted = torch.max(logits, 1)
            corrects += torch.sum(predicted == label)
            total_samples += label.size(0)

            print(loss.item())

            loss.backward()
            optimizer.step()
            global_counter += 1

            # losses.update(loss.data.item(), img.size()[0])
            # batch_time.update(time.time() - end)
            end = time.time()
            
            total_loss = total_loss + loss.item()
            # print(total_loss)
            if idx % args.disp_interval == args.disp_interval - 1:    # print every X mini-batches
                print('[Epoch %d, step %4d] loss: %.3f, corrects: %d, acc: %.3f' %
                    (current_epoch + 1, idx + 1, total_loss/args.disp_interval, corrects, 1.0*corrects/total_samples))
                neptune.log_metric('train_loss', total_loss/args.disp_interval)
                neptune.log_metric('lr', optimizer.param_groups[0]['lr'])
                total_loss = 0.0

        average_accuracy = 100 * corrects / total_samples
        average_loss = total_loss / total_samples

        val_loss = validate(model, val_loader, loss, criterion, total_samples)
        scheduler.step(val_loss)
        neptune.log_metric('epoch_train_acc', average_accuracy)
        neptune.log_metric('epoch_train_avg_loss', average_loss)
        print('Avg accuracy: ', average_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                            {'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            },
                            filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
            
        if current_epoch == total_epoch - 1:
            save_checkpoint(
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            },
                            filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
                
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
