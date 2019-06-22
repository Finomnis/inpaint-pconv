from torch import nn
from torchvision import models
import torch


class VGGExtractor(nn.Module):

    def __init__(self):
        super(VGGExtractor, self).__init__()

        self.vgg = models.vgg16(pretrained=True)

        # disable training for this network
        for param in self.vgg.parameters():
            param.requires_grad = False

        # extract the necessary layers
        self.stages = [
            self.vgg.features[0:5],
            self.vgg.features[5:10],
            self.vgg.features[10:17]
        ]

    def forward(self, img):
        results = list()

        for stage in self.stages:
            img = stage(img)
            results.append(img)

        return tuple(results)


class Lhole(nn.Module):
    def __init__(self):
        super(Lhole, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, i_gt, i_out, m):
        m_inverse = 1.0-m
        return self.l1(m_inverse * i_out, m_inverse * i_gt)


class Lvalid(nn.Module):
    def __init__(self):
        super(Lvalid, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, i_gt, i_out, mask):
        return self.l1(mask * i_out, mask * i_gt)


class Lperceptual(nn.Module):
    def __init__(self):
        super(Lperceptual, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, vgg_gt, vgg_out, vgg_comp):

        loss_out = [self.l1(a, b) for a, b in zip(vgg_out, vgg_gt)]
        loss_comp = [self.l1(a, b) for a, b in zip(vgg_comp, vgg_gt)]

        return torch.stack(tuple(loss_out+loss_comp)).sum()


class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, inp):
        # print(inp.size())
        b, c, h, w = inp.size()
        inp = inp.view(b, c, w*h)
        inp_t = inp.transpose(1, 2)

        # print(inp.size())
        # print(inp_t.size())
        gram = torch.bmm(inp, inp_t)

        k_p = 1.0/(c*w*h)
        gram = gram*k_p
        # print(gram.size())

        return gram


class Lstyle(nn.Module):
    def __init__(self):
        super(Lstyle, self).__init__()
        self.l1 = nn.L1Loss()
        self.gram = GramMatrix()

    def forward(self, vgg_gt, vgg_out, vgg_comp):

        gram_gt = [self.gram(el) for el in vgg_gt]

        loss_out = [self.l1(self.gram(el_out), el_gram_gt) for el_out, el_gram_gt in zip(vgg_out, gram_gt)]
        loss_comp = [self.l1(self.gram(el_comp), el_gram_gt) for el_comp, el_gram_gt in zip(vgg_comp, gram_gt)]

        return torch.stack(tuple(loss_out+loss_comp)).sum()


class Ltv(nn.Module):
    def __init__(self):
        super(Ltv, self).__init__()

    def forward(self, img, mask):

        b, c, h, w = img.size()

        diff_hor = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs_()
        diff_ver = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs_()

        mask_hor = 1 - mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_ver = 1 - mask[:, :, 1:, :] * mask[:, :, :-1, :]

        diff_hor *= mask_hor
        diff_ver *= mask_ver

        norm = 1.0/(c*h*w)
        loss_hor = norm * (diff_hor * mask_hor).sum()
        loss_ver = norm * (diff_ver * mask_ver).sum()

        # from DataLoader import MaskedImageDataset
        # MaskedImageDataset.to_img(diff_hor).show()
        # MaskedImageDataset.to_img(diff_ver).show()
        # MaskedImageDataset.to_img(mask_hor, True).show()
        # MaskedImageDataset.to_img(mask_ver, True).show()

        return loss_hor + loss_ver


class InpaintLoss(nn.Module):

    def __init__(self):
        super(InpaintLoss, self).__init__()

        self.l_valid = Lvalid()
        self.l_hole = Lhole()
        self.l_perceptual = Lperceptual()
        self.l_style = Lstyle()
        self.l_tv = Ltv()
        self.vgg_extractor = VGGExtractor()

    def forward(self, img_real, img_fake, img_comp, mask):

        vgg_real = self.vgg_extractor(img_real)
        vgg_fake = self.vgg_extractor(img_fake)
        vgg_comp = self.vgg_extractor(img_comp)

        loss_dict = {
            'valid': 1.0 * self.l_valid(img_real, img_fake, mask),
            'hole': 6.0 * self.l_hole(img_real, img_fake, mask),
            'perceptual': 0.05 * self.l_perceptual(vgg_real, vgg_fake, vgg_comp),
            'style': 120.0 * self.l_style(vgg_real, vgg_fake, vgg_comp),
            'tv': 0.1 * self.l_tv(img_comp, mask),
        }

        loss_dict['total'] = torch.stack(tuple(loss_dict.values())).sum()
        return loss_dict



# For testing purposes
def _main():

    from DataLoaders import MaskedImageDataset
    from torch.utils.data import DataLoader
    dataset = MaskedImageDataset('../datasets/places2/test_large')
    dataloader = DataLoader(dataset)

    device = torch.device('cuda')
    print(device)

    vgg_extractor = VGGExtractor().to(device)

    img_in, img_gt, mask = next(iter(dataloader))
    img_in = torch.nn.Parameter(img_in).to(device)
    img_gt = torch.nn.Parameter(img_gt).to(device)
    mask = torch.nn.Parameter(mask).to(device)

    img_out = img_in
    img_comp = mask * img_gt + (1.0-mask) * img_out

    vgg_gt = vgg_extractor(img_gt)
    vgg_out = vgg_extractor(img_out)
    vgg_comp = vgg_extractor(img_comp)

    print(img_in.requires_grad)
    l_perceptual = Lperceptual().to(device)
    l_style = Lstyle().to(device)
    print(l_perceptual(vgg_gt, vgg_out, vgg_comp))
    print(l_style(vgg_gt, vgg_out, vgg_comp))

    l_tv = Ltv().to(device)
    print(l_tv(img_comp, mask))

    loss = InpaintLoss().to(device)
    print(loss(img_gt, img_out, img_comp, mask))
    print(loss(img_gt, img_out, img_comp, mask))

    loss_dict = loss(img_gt, img_out, img_comp, mask)
    print(loss_dict)

    print(loss_dict['total'])

    from utils.dot import make_dot
    make_dot(loss_dict['total']).render('losses_graph', cleanup=True, view=True)


if __name__ == "__main__":
    _main()
