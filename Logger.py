import os
import time

import DataLoaders

class Logger:
    def __init__(self, save_dir):
        self.path = save_dir
        self.log_loss_msg('================ Training Loss (' + time.ctime() + ') ================')
        self.web_path = os.path.join(self.path, 'web')

        self.assert_web_initialized()

    def log_loss(self, iteration, epoch_size, loss_dict, time_taken, fine_tune):
        epoch, epoch_iteration = divmod(iteration, epoch_size)
        epoch += 1

        msg = '(epoch: ' + str(epoch) + ', iters: ' + str(epoch_iteration) + ', '
        msg += 'time: ' + '{:.3f}'.format(time_taken) + ', fine_tune: ' + str(fine_tune) + ')'

        for key in sorted(loss_dict):
            msg += ' ' + key + ': ' + '{:.3f}'.format(loss_dict[key])

        self.log_loss_msg(msg)

    def log_loss_msg(self, msg):
        print(msg)
        with open(os.path.join(self.path, 'loss_log.txt'), 'a') as fil:
            fil.write(msg+'\n')

    def assert_web_initialized(self):
        if not os.path.isdir(self.web_path):
            os.makedirs(self.web_path)

    def update_imgs(self, img_real, img_fake, img_comp, mask):
        img_real = DataLoaders.MaskedImageDataset.to_img(img_real[0])
        img_fake = DataLoaders.MaskedImageDataset.to_img(img_fake[0])
        img_comp = DataLoaders.MaskedImageDataset.to_img(img_comp[0])
        mask = DataLoaders.MaskedImageDataset.to_img(mask[0])

        img_real.save(os.path.join(self.web_path, 'train_real.png'))
        img_fake.save(os.path.join(self.web_path, 'train_fake.png'))
        img_comp.save(os.path.join(self.web_path, 'train_comp.png'))
        mask.save(os.path.join(self.web_path, 'train_mask.png'))

