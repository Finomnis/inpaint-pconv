
from torch import nn
import torch

from ext.partialconv.models.partialconv2d import PartialConv2d
from ext.partialconvtranspose2d import PartialConvTranspose2d

import warnings
import os
import datetime
import time

from utils.add_path import add_path

import Logger

with add_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ext', 'pix2pix')):
    from models.networks import init_weights


# Important:
# All layers return both data and mask


class PartialNorm(nn.BatchNorm2d):

    def __init__(self, *args, **kwargs):
        super(PartialNorm, self).__init__(*args, **kwargs)

    def forward(self, x_in, mask):
        x_out = super(PartialNorm, self).forward(x_in)
        return x_out, mask


class PartialLeakyRelu(nn.LeakyReLU):

    def __init__(self):
        super(PartialLeakyRelu, self).__init__(0.2, True)

    def forward(self, x_in, mask):
        x_out = super(PartialLeakyRelu, self).forward(x_in)
        return x_out, mask


class PartialRelu(nn.ReLU):

    def __init__(self):
        super(PartialRelu, self).__init__(True)

    def forward(self, x_in, mask):
        x_out = super(PartialRelu, self).forward(x_in)
        return x_out, mask


class PartialTanh(nn.Tanh):
    def __init__(self):
        super(PartialTanh, self).__init__()

    def forward(self, x_in, mask):
        x_out = super(PartialTanh, self).forward(x_in)
        return x_out, mask


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64, fine_tune=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, innermost=True, fine_tune=fine_tune)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, fine_tune=fine_tune)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, fine_tune=fine_tune)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, fine_tune=fine_tune)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, fine_tune=fine_tune)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, submodule=unet_block, outermost=True, fine_tune=fine_tune)  # add the outermost layer

    def forward(self, inp, mask):
        """Standard forward"""
        result_img, result_mask = self.model(inp, mask)
        result_comp = mask * inp + (1.0-mask) * result_img
        return result_img, result_comp, result_mask


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, fine_tune=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = PartialConv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1,
                                 multi_channel=True,
                                 return_mask=True)
        uprelu = PartialRelu()

        if outermost:
            upconv = PartialConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1,
                                            multi_channel=True,
                                            return_mask=True)
            down = [downconv]
            up = [uprelu, upconv, PartialTanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = PartialConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1,
                                            multi_channel=True,
                                            return_mask=True)
            down = [PartialLeakyRelu(), downconv]
            up = [uprelu, upconv, PartialNorm(outer_nc)]
            model = down + up
        else:
            upconv = PartialConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1,
                                            multi_channel=True,
                                            return_mask=True)
            down_norm = PartialNorm(inner_nc)
            # Disable training for encoder BatchNorm at fine-tuning step
            if fine_tune:
                print('Disabling parameters for down_norm ...')
                for param in down_norm.parameters():
                    param.requires_grad = False
            down = [PartialLeakyRelu(), downconv, down_norm]
            up = [uprelu, upconv, PartialNorm(outer_nc)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x_in, mask_in):
        x_out, mask_out = x_in, mask_in
        for layer in self.model:
            x_out, mask_out = layer(x_out, mask_out)

        if self.outermost:
            return x_out, mask_out
        else:
            return torch.cat([x_in, x_out], 1), torch.cat([mask_in, mask_out], 1)


def init_net(net, init_type=None, init_gain=0.02):

    # Get number of cuda devices
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))

    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    else:
        warnings.warn('No GPU detected! Network will run on CPU!')

    if init_type is not None:
        init_weights(net, init_type, init_gain)
    return net


def endless_iterator(generator):
    while True:
        for i in generator:
            yield i


class PConvInfilNet():

    def __init__(self, model_save_path, load_weights='best', fine_tune=False):

        self.model_save_path = model_save_path
        self.iteration = 0

        if load_weights:
            self.model = init_net(UnetGenerator(3, 3, num_downs=8, ngf=64, fine_tune=fine_tune), init_type=None)
            self.load_params(load_weights)
        else:
            self.model = init_net(UnetGenerator(3, 3, num_downs=8, ngf=64, fine_tune=fine_tune), init_type='kaiming')

        # Training parameters
        self.fine_tune = fine_tune
        self.optim = None
        self.train_data = None
        self.val_data = None
        self.vis_data = None
        self.log_interval = None
        self.save_interval = None
        self.epoch_size = None


    def load_params(self, network_type):
        save_dict = torch.load(os.path.join(self.model_save_path, 'latest_net_' + network_type + '.pth'))
        self.model.load_state_dict(save_dict['network_params'])
        self.iteration = save_dict['iteration']

    def save_params(self, network_type):
        save_dict = dict()
        save_dict['network_params'] = self.model.state_dict()
        save_dict['iteration'] = self.iteration
        torch.save(os.path.join(self.model_save_path, 'latest_net_' + network_type + '.pth'))

    def get_params(self):
        return self.model.parameters()

    def set_training_params(self, optim, loss_func, train_data, val_data, vis_data, log_interval, save_interval):
        self.optim = optim
        self.loss_func = loss_func
        self.train_data = train_data
        self.val_data = val_data
        self.vis_data = vis_data
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.epoch_size = len(train_data)

        self.train_iter = endless_iterator(self.train_data)
        self.val_iter = endless_iterator(self.val_data)

        print('=== Optimizer ===')
        print(self.optim)
        print()
        print(' - epoch size:    ', self.epoch_size)
        print(' - log interval:  ', self.log_interval)
        print(' - save interval: ', self.save_interval)

        print(vis_data[0].size())

        self.logger = Logger.Logger(self.model_save_path)

    def train_step(self):
        t0 = time.perf_counter()
        img_in, img_real, mask = next(self.train_iter)

        # forward
        t1 = time.perf_counter()
        img_fake, img_comp, mask_fake = self.forward(img_in, mask)

        # loss
        t2 = time.perf_counter()
        loss_dict = self.loss_func(img_real, img_fake, img_comp, mask)
        loss = loss_dict['total']

        # optimize
        t3 = time.perf_counter()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        t4 = time.perf_counter()

        # logs
        batch_size =  img_real.size(0)
        old_iteration = self.iteration
        self.iteration = old_iteration + batch_size
        if self.iteration % self.log_interval <= old_iteration % self.log_interval:
            self.logger.log_loss(self.iteration, self.epoch_size, loss_dict, (t4-t0) / batch_size, self.fine_tune)
            self.logger.update_imgs(img_real, img_fake, img_comp, mask)

        if self.iteration % self.save_interval <= old_iteration % self.save_interval:
            self.save_all()

    def forward(self, img, mask):
        return self.model(img, mask)
