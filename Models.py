
from torch import nn
import torch

from ext.partialconv.models.partialconv2d import PartialConv2d
from ext.partialconvtranspose2d import PartialConvTranspose2d

import warnings
import os
from utils.add_path import add_path

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

    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64):
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
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, submodule=unet_block, outermost=True)  # add the outermost layer

    def forward(self, inp, mask):
        """Standard forward"""
        return self.model(inp, mask)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
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
            down = [PartialLeakyRelu(), downconv, PartialNorm(inner_nc)]
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


def init_net(net, init_type='kaiming', init_gain=0.02):

    # Get number of cuda devices
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))

    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    else:
        warnings.warn('No GPU detected! Network will run on CPU!')

    init_weights(net, init_type, init_gain)
    return net
