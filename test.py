from DataLoader import MaskedImageDataset
from Models import UnetGenerator

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def main():
    dataset = MaskedImageDataset('/home/martin/Downloads/test_large')
    dataloader = DataLoader(dataset)

    print(len(dataset.imgs))
    print(len(dataset.masks))

    it = iter(dataloader)

    print(next(it))

    model = UnetGenerator(3, 3)

    img, img_gt, mask = next(it)

    result, result_mask = model(img, mask)

    plt.figure('Stencil')
    plt.subplot(2, 3, 1)
    plt.imshow(MaskedImageDataset.to_img(img))
    plt.subplot(2, 3, 2)
    plt.imshow(MaskedImageDataset.to_img(mask, True))
    plt.subplot(2, 3, 3)
    plt.imshow(MaskedImageDataset.to_img(img_gt))
    plt.subplot(2, 3, 4)
    plt.imshow(MaskedImageDataset.to_img(result))
    plt.subplot(2, 3, 5)
    plt.imshow(MaskedImageDataset.to_img(result_mask, is_mask=True))
    plt.show()


if __name__ == "__main__":
    main()
