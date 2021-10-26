import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np

class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, img_root, label_root, transform, t_transform, filename=None):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + '.jpg'), self.image_path))
        else:
            lines = [(line.rstrip('\n')[:-2], line.rstrip('\n')[-1])  for line in open(filename)]
            self.names = list(map(lambda x: os.path.join(x[0].split(' ')[0]), lines))
            self.image_path = list(map(lambda x: os.path.join(img_root, x[0]), lines))
            self.label_path = list(map(lambda x: os.path.join(label_root, x[0]), lines))
            self.labels = list(map(lambda x: os.path.join('', x[1]), lines))
        # print(self.image_path[0])
        # print(self.label_path[0])
        # print(self.names[0])
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        if self.labels[item]=='7':
            label = Image.open(self.label_path[item]).convert('L')
        else:
            label = Image.new('RGB', image.size)
            label = label.convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label, self.names[item]

    def __len__(self):
        return len(self.image_path)



# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, filename=None, mode='train', num_thread=2, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, transform, t_transform, filename=filename)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, None, t_transform, filename=filename)
        return dataset