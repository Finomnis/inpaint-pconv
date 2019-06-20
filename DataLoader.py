from torch.utils.data import Dataset
import torch
from torchvision import transforms

from PIL import Image

import numpy as np

import os
import glob
import random

class MaskedImageDataset(Dataset):

    mean = [0.485, 0.456, 0.406]
    stddev = [0.229, 0.224, 0.225]

    @staticmethod
    def to_img(x):
        x.transpose_(0, 2)
        x = x * torch.Tensor(MaskedImageDataset.stddev) + torch.Tensor(MaskedImageDataset.mean)

        if x.size(2) == 3:
            img = Image.fromarray(x.cpu().detach().numpy(), 'RGB')
        else:
            img = Image.fromarray(x.cpu().detach().numpy(), 'L')

        return img

    def __init__(self, img_path, mask_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'masks', 'generated')):
        super(MaskedImageDataset, self).__init__()

        self.imgs = glob.glob(img_path + '**/*.jpg', recursive=True)
        self.masks = glob.glob(mask_path + '**/*.png')

        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(MaskedImageDataset.mean, MaskedImageDataset.stddev)])
        self.mask_transform = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')

        crop_size = min(img.size)
        img = transforms.RandomCrop((crop_size, crop_size))(img)
        if crop_size < 512:
            img = transforms.Resize((512, 512))(img)

        img = self.img_transform(img)

        mask = Image.open(random.choice(self.masks)).convert('RGB')
        mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
