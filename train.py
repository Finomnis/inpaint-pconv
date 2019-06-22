import argparse
import os

from Logger import Logger
import DataLoaders
import Models
import Losses

import torch
from torch.utils.data import DataLoader


def main():
    file_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help='The training set. Should mostly have side length >512px. Recommended: ')
    parser.add_argument("val_path", help='The validation set. Should mostly have side length >512px.')
    parser.add_argument("--mask_path",
                        default=os.path.join(file_path, 'masks', 'generated'),
                        help='The path to the mask images. Need to be 512x512.')

    parser.add_argument("--checkpoint_dir", default=os.path.join(file_path, 'checkpoints'),
                        help='The checkpoint directory')
    parser.add_argument("--model_name", default="test")
    parser.add_argument("--continue_train", action='store_true')

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--vis_interval", type=int, default=5000)
    parser.add_argument("--stage_interval", type=int, default=50000)
    args = parser.parse_args()

    print(args)

    # Create Model
    model_save_path = os.path.join(args.checkpoint_dir, args.model_name)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    model = Models.PConvInfilNet(model_save_path, load_weights='latest' if args.continue_train else None, fine_tune=args.fine_tune)

    # Open Datasets
    train_dataset = DataLoaders.MaskedImageDataset(args.img_path, mask_path=args.mask_path)
    val_dataset = DataLoaders.MaskedImageDataset(args.val_path, mask_path=args.mask_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # Initialize Optimizer
    optim = torch.optim.Adam(model.get_params(), lr=(args.fine_tune_lr if args.fine_tune else args.lr))

    # Initialize Loss
    loss = Models.init_net(Losses.InpaintLoss(), init_type=None)

    # Send parameters to model
    model.set_training_params(optim,
                              loss,
                              train_dataloader,
                              val_dataloader,
                              val_dataset.get_examples(args.batch_size),
                              args.log_interval,
                              args.save_interval,
                              args.vis_interval,
                              args.stage_interval)

    while True:
        model.train_step()


if __name__ == '__main__':
    main()