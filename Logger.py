import os
import time

import DataLoaders

html_file = """
<!DOCTYPE html><head><title>Training</title></head><body>
    <table>
        <tr>
            <th></th>
            <th>Real</th>
            <th>Fake</th>
            <th>Comp</th>
            <th>Mask</th>
        </tr>
        <tr>
            <th>Train</th>
            <td><img id="train_real" src="train_real.png"/></td>
            <td><img id="train_fake" src="train_fake.png"/></td>
            <td><img id="train_comp" src="train_comp.png"/></td>
            <td><img id="train_mask" src="train_mask.png"/></td>
        </tr>
        <tr>
            <th>Test</th>
            <td><img id="val_real" src="val_real.png"/></td>
            <td><img id="val_fake" src="val_fake.png"/></td>
            <td><img id="val_comp" src="val_comp.png"/></td>
            <td><img id="val_mask" src="val_mask.png"/></td>
        </tr>
        <tr>
            <th>Visual</th>
            <td><img id="vis_real" src="vis/img_real_0.png"/></td>
            <td><img id="vis_fake" src="vis/img_fake_0.png"/></td>
            <td><img id="vis_comp" src="vis/img_comp_0.png"/></td>
            <td><img id="vis_mask" src="vis/img_mask_0.png"/></td>
        </tr>
    </table>    
    <script>
    (function(){
        let date = new Date().getTime();
        let img_ids = ["real", "fake", "comp", "mask"];
        for(let img_id of img_ids){
            document.getElementById('train_'+img_id).src='train_'+img_id+".png?t="+date;
            document.getElementById('val_'+img_id).src='val_'+img_id+".png?t="+date;
            document.getElementById('vis_'+img_id).src='vis/img_'+img_id+"_0.png?t="+date;
        }
        setTimeout(arguments.callee, 30000);
    })();
    </script>
    <!-- page content -->
</body></html>
"""


class Logger:
    def __init__(self, save_dir):
        self.path = save_dir
        self.log_loss_msg('================ Training Loss (' + time.ctime() + ') ================')
        self.log_info_msg('')
        self.log_info_msg('================ Training Info (' + time.ctime() + ') ================')
        self.web_path = os.path.join(self.path, 'web')
        self.vis_path = os.path.join(self.web_path, 'vis')

        self.assert_web_initialized()

    def log_loss(self, iteration, epoch_size, loss_dict, loss_dict_val, time_taken, fine_tune):
        epoch, epoch_iteration = divmod(iteration, epoch_size)
        epoch += 1

        msg = '(epoch: ' + str(epoch) + ', iters: ' + str(epoch_iteration) + ', '
        msg += 'time: ' + '{:.3f}'.format(time_taken) + ', fine_tune: ' + str(fine_tune) + ')'

        for key in sorted(loss_dict):
            msg += ' train_' + key + ': ' + '{:.3f}'.format(loss_dict[key].item())
        for key in sorted(loss_dict_val):
            msg += ' val_' + key + ': ' + '{:.3f}'.format(loss_dict_val[key].item())

        self.log_loss_msg(msg)

    def log_info_msg(self, *args):
        msg = " ".join([str(arg) for arg in args])
        print(msg)
        with open(os.path.join(self.path, 'info_log.txt'), 'a') as fil:
            fil.write(msg+'\n')

    def log_loss_msg(self, msg):
        print(msg)
        with open(os.path.join(self.path, 'loss_log.txt'), 'a') as fil:
            fil.write(msg+'\n')

    def create_folder_if_missing(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)

    def assert_web_initialized(self):
        self.create_folder_if_missing(self.web_path)
        self.create_folder_if_missing(self.vis_path)
        with open(os.path.join(self.web_path, 'index.html'), 'w') as fil:
            fil.write(html_file)

    def update_imgs(self, img_real, img_fake, img_comp, mask, val_real, val_fake, val_comp, val_mask):
        img_real = DataLoaders.MaskedImageDataset.to_img(img_real[0])
        img_fake = DataLoaders.MaskedImageDataset.to_img(img_fake[0])
        img_comp = DataLoaders.MaskedImageDataset.to_img(img_comp[0])
        mask = DataLoaders.MaskedImageDataset.to_img(mask[0])

        img_real.save(os.path.join(self.web_path, 'train_real.png'))
        img_fake.save(os.path.join(self.web_path, 'train_fake.png'))
        img_comp.save(os.path.join(self.web_path, 'train_comp.png'))
        mask.save(os.path.join(self.web_path, 'train_mask.png'))

        val_real = DataLoaders.MaskedImageDataset.to_img(val_real[0])
        val_fake = DataLoaders.MaskedImageDataset.to_img(val_fake[0])
        val_comp = DataLoaders.MaskedImageDataset.to_img(val_comp[0])
        val_mask = DataLoaders.MaskedImageDataset.to_img(val_mask[0])

        val_real.save(os.path.join(self.web_path, 'val_real.png'))
        val_fake.save(os.path.join(self.web_path, 'val_fake.png'))
        val_comp.save(os.path.join(self.web_path, 'val_comp.png'))
        val_mask.save(os.path.join(self.web_path, 'val_mask.png'))

    def update_visualization(self, iter, imgs_real, imgs_fake, imgs_comp, imgs_mask):
        for i, (img_real, img_fake, img_comp, img_mask) in enumerate(zip(imgs_real, imgs_fake, imgs_comp, imgs_mask)):

            img_real = DataLoaders.MaskedImageDataset.to_img(img_real)
            img_fake = DataLoaders.MaskedImageDataset.to_img(img_fake)
            img_comp = DataLoaders.MaskedImageDataset.to_img(img_comp)
            img_mask = DataLoaders.MaskedImageDataset.to_img(img_mask)

            img_real.save(os.path.join(self.vis_path, 'img_real_' + str(i) + '.png'))
            img_fake.save(os.path.join(self.vis_path, 'img_fake_' + str(i) + '.png'))
            img_comp.save(os.path.join(self.vis_path, 'img_comp_' + str(i) + '.png'))
            img_mask.save(os.path.join(self.vis_path, 'img_mask_' + str(i) + '.png'))

            img_log_folder = os.path.join(self.vis_path, str(i))
            self.create_folder_if_missing(img_log_folder)
            img_fake.save(os.path.join(img_log_folder, "{:09d}".format(iter) + '.png'))

