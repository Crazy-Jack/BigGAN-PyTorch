import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
from utils_imgpool import ImagePool


# for gradient ascent
class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if targeted:
            cost = self.criterion(h_adv, y)
        else:
            cost = -self.criterion(h_adv, y)

        # self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        # print(f'L-inf norm of perturbation  {torch.max(torch.abs(eps*x_adv.grad)).item()}')
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        h = self.net(x)    # original
        h_adv = self.net(x_adv)     # after attack
        print(f'Before attack, {self.criterion}:    {self.criterion(h, y):.6f}')
        print(f'After attack, {self.criterion}:    {self.criterion(h_adv, y):.6f}')

        return x_adv, h_adv, h


def evaluate_sample(config, fixed_x, fixed_y, G, E, experiment_name, attack=False, sample_randomly=False):
    with torch.no_grad():
        fixed_z, _, _ = E(fixed_x)
        fixed_z = fixed_z.detach()
        print("fixed_x: ", fixed_x.shape)
        print("fixed_z: ", fixed_z.shape)
        print("fixed_y: ", fixed_y.shape)
        output = G(fixed_z, G.shared(fixed_y), 0, return_inter_activation=True)

        fixed_Gz = output[0].detach()

        n, c, h, w = fixed_Gz.shape
        # # log masks
        save_dir = '%s/%s' % (config['samples_root'], experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        image_filename = save_dir + '/fixed_samples.jpg'
        fixed_Gz = torch.tensor(fixed_Gz)
        torchvision.utils.save_image(fixed_Gz, image_filename,
                                     nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)

    if attack:
        def net_wrapper(z):
            out = G(z, G.shared(fixed_y[:12]), 0, return_inter_activation=False)[0]
            return out
        criterion = nn.L1Loss()
        attacker = Attack(net_wrapper, criterion)
        z_adv, Gz_adv, _ = attacker.fgsm(fixed_z[:12], fixed_x[:12])
        Gz_adv = torch.tensor(Gz_adv.detach())
        attack_image_filename = save_dir + '/attacked_samples.jpg'
        attack_gt_filename = save_dir + '/attacked_gt.jpg'
        
        torchvision.utils.save_image(fixed_x[:12], attack_gt_filename,
                                     nrow=int(fixed_x[:12].shape[0] ** 0.5), normalize=True)

        torchvision.utils.save_image(Gz_adv, attack_image_filename,
                                     nrow=int(Gz_adv.shape[0] ** 0.5), normalize=True)


def evaluate_metrics(dataloader, E, G, sample_dir, device, num_inception_imgs=50000):
    # generate enough imgs
    # then use pytorch_fid to compute fid
    # python -m pytorch_fid path/to/dataset1 path/to/dataset2
    # can be installed via pip install pytorch_fid
    pbar = tqdm(dataloader[0])
    sample_count = 0
    os.makedirs(sample_dir, exist_ok=True)

    for i, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            z, _, _ = E(x)
            output = G(z, G.shared(y), 0, return_inter_activation=False)
            generated_sample = output[0].detach()
            generated_sample = torch.tensor(generated_sample)

        for image_idx in range(generated_sample.size(0)):
            torchvision.utils.save_image(generated_sample[image_idx, :, :, :],
                                         os.path.join(sample_dir, f'{sample_count+image_idx}.jpg'),
                                         normalize=True)

        sample_count += generated_sample.size(0)
        if sample_count > num_inception_imgs:
            break


def run(config):
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)
    E = model.ImgEncoder(**config).to(device)
    GDE = model.G_D_E(G, D, E)

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    print('Number of params in G: {} D: {} E: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D, E]]))

    print('Loading weights...')
    utils.load_weights(G, D, E, state_dict,
                       config['weights_root'], experiment_name,
                       config['load_weights'] if config['load_weights'] else None,
                       None, strict=False, load_optim=False)

    # ==============================================================================
    # prepare the data
    loaders, train_dataset = utils.get_data_loaders(**config)

    G_batch_size = max(config['G_batch_size'], config['batch_size'])

    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                              device=device)

    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device)

    fixed_z.sample_()
    fixed_y.sample_()
    print("fixed_y original: {} {}".format(fixed_y.shape, fixed_y[:10]))

    fixed_x, fixed_y_of_x = utils.prepare_x_y(G_batch_size, train_dataset, experiment_name, config)

    evaluate_sample(config, fixed_x, fixed_y, G, E, experiment_name, attack=True)
    # evaluate_metrics(loaders, E, G, 'generated_samples', device)


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    # print(config)
    run(config)


if __name__ == '__main__':
    main()