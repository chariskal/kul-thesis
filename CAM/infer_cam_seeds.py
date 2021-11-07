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

image_file = '/home/charis/kul-thesis/kvasir-dataset-v2/images/polyps/cju0qkwl35piu0993l0dewei2.jpg'

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


model = models.resnet50(pretrained=True)
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

finalconv_name = 'layer4'


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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_FINETUNE)

model, optimizer, _, _ = load_ckp(os.path.join(SNAPSHOT_DIR, 'kvasir_epoch_35.pth'), model, optimizer)

# hook the feature extractor
net = model.cpu()
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)


h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)