from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch
from torch import nn
import os
import cv2
import json
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import shutil

import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# input image
BASE_PATH = "/home/charis/kul-thesis/kvasir-dataset-v2/"
BASE_PATH = '/home/charis/kul-thesis/kvasir-dataset-v2-folds/1/'
VAL_SPLIT = 0.1
TRAIN = os.path.join(BASE_PATH, "train")
TEST = os.path.join(BASE_PATH, "test")
LABELS_file = 'kvasir-labels.json'
SNAPSHOT_DIR = 'checkpoints/train/exp1'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
FEATURE_EXTRACTION_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 2
PRED_BATCH_SIZE = 8
EPOCHS = 40
LR = 0.001
LR_FINETUNE = 0.0005
disp_interval = 400

if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

def save_checkpoint(state, filename='checkpoint.pth.tar'):     # save points during training
    """Function for saving checkpoints"""
    savepath = os.path.join(SNAPSHOT_DIR, filename)
    torch.save(state, savepath)
    print(f"model saved to {savepath}")

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']+1, checkpoint['global_counter']

def validate(model, criterion):
    model.eval()
    corrects = 0
    total_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for idx, dat in enumerate(val_loader):
            imgs, labels = dat
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            corrects += (predicted == labels).sum().item()
            total_loss += loss.item()

    average_accuracy = 100 * corrects / total_samples
    average_loss = total_loss / total_samples
    return average_loss, average_accuracy

# modelworks such as googlemodel, resmodel, densemodel already use global average pooling at the end, so CAM could be used directly.

model = models.resnet50(pretrained=True)
finalconv_name = 'layer4'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dataset = 'kvasir'

# load the imagemodel category list
with open(LABELS_file) as f:
    classes = json.load(f)

numFeatures = model.fc.in_features
# loop over the modules of the model and set the parameters of
# batch normalization modules as not trainable
for module, param in zip(model.modules(), model.parameters()):
	if isinstance(module, nn.BatchNorm2d):
		param.requires_grad = False
# define the modelwork head and attach it to the model
headmodel = nn.Sequential(
	nn.Linear(numFeatures, len(classes)),
)
model.fc = headmodel
# append a new classification top to our feature extractor and pop it
# on to the current device
model = model.to(device)


# specify Imagemodel mean and standard deviation and image size


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, batch_size=FINETUNE_BATCH_SIZE, mode='train', num_thread=os.cpu_count(), pin=True):
    if mode == 'train':
        # define augmentation pipelines
        transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        dataset = datasets.ImageFolder(root=img_root, transform=transform)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return dataset, data_loader
    else:
        t_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        dataset = datasets.ImageFolder(root=img_root, transform=t_transform)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                      pin_memory=pin)
        return dataset, data_loader

train_dataset, train_loader = get_loader(img_root=TRAIN)
val_dataset, val_loader = get_loader(img_root=TEST, mode='val')

# initialize loss function and optimizer (notice that we are only
# providing the parameters of the classification top to our optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_FINETUNE)
# calculate steps per epoch for training and validation set
trainSteps = len(train_dataset) // FEATURE_EXTRACTION_BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": []}

print("[INFO] training the NETwork...")
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
model = model.to(device)

global_counter = 0
best_val_loss = float('inf')
current_epoch = 0

while current_epoch < EPOCHS:
    model.train()
    total_loss = 0
    corrects = 0
    total_samples = 0

    for idx, dat in tqdm(enumerate(train_loader, 0)):
        # send the input to the device
        inputs, labels = dat
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        corrects += torch.sum(predicted == labels)
        total_samples += labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        global_counter += 1

        # print statistics
        total_loss += loss.item()
        if idx % disp_interval == disp_interval-1:    # print every X mini-batches
            print('[%d, %5d] loss: %.3f, corrects: %d' %
                  (current_epoch + 1, idx + 1, total_loss/disp_interval, corrects))
            total_loss = 0.0
    average_accuracy = 100 * corrects / total_samples
    average_loss = total_loss / (1.0*total_samples)

    print(f"Training Loss: {average_loss:.6f} at epoch: {current_epoch}")
    print(f"Training Accuracy: {average_accuracy:.4f} %  at epoch: {current_epoch}")
    lr = optimizer.param_groups[0]["lr"]
    print(f"Learning rate: {lr}  at epoch: {current_epoch}")

    val_loss, val_acc = validate(model, criterion)
    print("-------------------------------------------------------")
    print(f"Validation Loss: {val_loss:.6f} at epoch: {current_epoch}")
    print(f"Validation Accuracy: {val_acc:.4f} %  at epoch: {current_epoch}")
    print(
        "-------------------------------------------------------\
        -------------------------------------------------------"
    )

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(
                      {'epoch': current_epoch,
                        'global_counter': global_counter,
                        'state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict()
                      },
                      filename='%s_epoch_%d.pth' %(dataset, current_epoch))
    
    if current_epoch == EPOCHS-1:
        save_checkpoint(
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        },
                        filename='%s_epoch_%d.pth' %(dataset, current_epoch))
        
    current_epoch += 1

    # update our training history
    # H["train_loss"].append(average_loss.cpu().detach().numpy())
    # H["train_acc"].append(corrects)
    # print the model training and validation information
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        average_loss, corrects))
