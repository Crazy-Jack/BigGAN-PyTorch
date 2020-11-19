""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

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
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback



def activation_extract(G, D, E, G_ema, fixed_x, fixed_y, z_, y_,
                    state_dict, config, experiment_name, save_weights=False):
    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    # Save a random sample sheet with fixed z and y
    # TODO: change here to encode fixed x into z and feed z with fixed y into G
    print("check2 ---------------------------")
    with torch.no_grad():
        if config['parallel']:
            print("fixed_x: ", fixed_x.shape)
            if config['inference_nosample']:
                _, fixed_z, _ = nn.parallel.data_parallel(E, fixed_x)
            else:
                fixed_z, _, _ =  nn.parallel.data_parallel(E, fixed_x)
            print("fixed_z: ", fixed_z.shape)
            print("fixed_y: ", fixed_y.shape)
            fixed_Gz, intermed_activation = nn.parallel.data_parallel(
                which_G, (fixed_z, which_G.shared(fixed_y), True))
        else:
            # print("fixed_x: ", fixed_x.shape)
            fixed_z, _, _ = E(fixed_x)
            # print("fixed_z: ", fixed_z.shape)
            # print("fixed_y: ", fixed_y.shape)
            #### TODO: Catch the intermediate output here AND save the results into numpy
            fixed_Gz, intermed_activation = which_G(fixed_z, which_G.shared(fixed_y), True)
    print("check3 -----------------------------")
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/test_iter_%s.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    image_origin_filename = '%s/%s/origin_iter_%s.jpg' % (config['samples_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    activation_filename = '%s/%s/inter_activation_iter_%s.npy' % (config['samples_root'],
                                                    experiment_name, state_dict['itr'])
    print("######### save_path #####: ", image_filename)
    torchvision.utils.save_image(fixed_x.float().cpu(), image_origin_filename,
                                 nrow=int(fixed_x.shape[0] ** 0.5), normalize=True)
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    np.save(activation_filename, intermed_activation)

    





def run(config):

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cpu'

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
    experiment_name = "test_{}".format(experiment_name)
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)
    E = model.ImgEncoder(**config).to(device)
    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(
            config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?
    GDE = model.G_D_E(G, D, E)
    # print(G)
    # print(D)
    # print(E)
    print("Model Created!")
    print('Number of params in G: {} D: {} E: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D, E]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights

    print('Loading weights...')
    utils.load_weights(G, D, E, state_dict,
                        config['weights_root'], config['load_experiment_name'],
                        config['load_weights'] if config['load_weights'] else None,
                        G_ema if config['ema'] else None)
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}
    # If parallel, parallelize the GD module
    if config['parallel']:
        GDE = nn.DataParallel(GDE)
        if config['cross_replica']:
            patch_replication_callback(GDE)

    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])
    loaders, train_dataset = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': 0})

    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    print("fixed_y original: {} {}".format(fixed_y.shape, fixed_y[:10]))
    fixed_x, fixed_y_of_x = utils.prepare_x_y(G_batch_size, train_dataset, experiment_name, config)


    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                   else G),
                               z_=z_, y_=y_, config=config)

    G.eval()
    E.eval()
    print("check1 -------------------------------")
    print("state_dict['itr']", state_dict['itr'])
    if config['pbar'] == 'mine':
        pbar = utils.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')

    else:
        pbar = tqdm(loaders[0])
    
    print("state_dict['itr']", state_dict['itr'])
    for i, (x, y) in enumerate(pbar):
        state_dict['itr'] += 1
        if config['D_fp16']:
            x, y = x.to(device).half(), y.to(device)
        else:
            x, y = x.to(device), y.to(device)
        print("x.shape", x.shape)
        print("y.shape", y.shape)
        
   
        activation_extract(G, D, E, G_ema, x, y, z_, y_,
                                          state_dict, config, experiment_name,
                                           save_weights=False)
        if state_dict['itr'] == 20:
            break



def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()

