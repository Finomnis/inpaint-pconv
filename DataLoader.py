from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import os
import glob
import random


class MaskedImageDataset(Dataset):

    mean = [0.485, 0.456, 0.406]
    stddev = [0.229, 0.224, 0.225]

    @staticmethod
    def unnormalize(img):
        pass

    def __init__(self, img_path, mask_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'masks', 'generated')):
        self.imgs = glob.glob(img_path + '**/*.jpg', recursive=True)
        self.masks = glob.glob(mask_path + '**/*.png')

        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(MaskedImageDataset.mean, MaskedImageDataset.stddev)])
        self.mask_transform = transforms.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        mask = Image.open(random.choice(self.masks)).convert('L')

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
