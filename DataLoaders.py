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
    def unnormalize(x, is_mask=False):
        x = x.detach().cpu()
        while len(x.size()) > 3:
            x = x.squeeze(0)

        x = x.permute(1, 2, 0)

        if not is_mask:
            x = x * torch.Tensor(MaskedImageDataset.stddev) + torch.Tensor(MaskedImageDataset.mean)
        x = x.detach().numpy()
        x = (x*255).round().clip(0, 255).astype(np.uint8)
        return x

    @staticmethod
    def to_img(x, is_mask=False):
        x = MaskedImageDataset.unnormalize(x, is_mask=is_mask)

        if x.shape[2] == 3:
            img = Image.fromarray(x, 'RGB')
        else:
            img = Image.fromarray(x, 'L')

        return img

    def __init__(self, img_path, mask_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'masks', 'generated')):
        super(MaskedImageDataset, self).__init__()

        self.imgs = glob.glob(img_path + '**/*.jpg', recursive=True)
        self.masks = glob.glob(mask_path + '**/*.png')

        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(MaskedImageDataset.mean, MaskedImageDataset.stddev)])
        self.mask_transform = transforms.ToTensor()

    def __getitem__(self, index, mask_id=None):
        img = Image.open(self.imgs[index]).convert('RGB')

        # Random crop/resize image to 512x512
        crop_size = min(512, min(img.size))
        img = transforms.RandomCrop((crop_size, crop_size))(img)
        if crop_size < 512:
            img = transforms.Resize((512, 512))(img)

        img = self.img_transform(img)

        if mask_id is None:
            mask = Image.open(random.choice(self.masks)).convert('RGB')
        else:
            mask = Image.open(self.masks[mask_id]).convert('RGB')
        mask = self.mask_transform(mask)

        return img*mask, img, mask

    def __len__(self):
        return len(self.imgs)

    def get_examples(self, num):
        example_imgs = list()
        example_masks = list()

        for i in range(num):
            comb, img, mask = self.__getitem__(i, i)
            example_imgs.append(img)
            example_masks.append(mask)

        return torch.stack(example_imgs), torch.stack(example_masks)
