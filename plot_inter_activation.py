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
from utils_imgpool import ImagePool



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
                _, fixed_z, _ = [i.detach() for i in nn.parallel.data_parallel(E, fixed_x)]
            else:
                fixed_z, _, _ =  [i.detach() for i in nn.parallel.data_parallel(E, fixed_x)]
            # fixed_z = fixed_z.detach()
            print("fixed_z: ", fixed_z.shape)
            print("fixed_y: ", fixed_y.shape)
        
            fixed_Gz, intermediates = nn.parallel.data_parallel(
                which_G, (fixed_z, which_G.shared(fixed_y), int(1 / config['sparse_decay_rate']) + 100, True)).detach()
            
            if (not config['no_adaptive_tau']) and (state_dict['itr'] * config['sparse_decay_rate'] < 1.1):
                fixed_Gz_train, intermediates = nn.parallel.data_parallel(
                    which_G, (fixed_z, which_G.shared(fixed_y), state_dict['itr'], True)).detach()
        else:
            fixed_z, _, _ = E(fixed_x)
            fixed_Gz, intermediates = which_G(fixed_z, which_G.shared(fixed_y), int(1 / config['sparse_decay_rate']) + 100, return_inter_activation=True)
            if (not config['no_adaptive_tau']) and (state_dict['itr'] * config['sparse_decay_rate'] < 1.1):
                fixed_Gz_train, intermediates = which_G(fixed_z, which_G.shared(fixed_y), state_dict['itr'], return_inter_activation=True)

    print("check3 -----------------------------")
    if not os.path.isdir('%s/%s' % (config['evals_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['evals_root'], experiment_name))
    image_filename = '%s/%s/test_iter_%s.jpg' % (config['evals_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    image_origin_filename = '%s/%s/origin_iter_%s.jpg' % (config['evals_root'],
                                                    experiment_name,
                                                    state_dict['itr'])
    activation_filename = '%s/%s/inter_activation_iter_%s.pt' % (config['evals_root'],
                                                    experiment_name, state_dict['itr'])
    print("######### save_path #####: ", image_filename)
    torchvision.utils.save_image(fixed_x.float().cpu(), image_origin_filename,
                                 nrow=int(fixed_x.shape[0] ** 0.5), normalize=True)
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    torch.save(intermediates, activation_filename)

    return fixed_x.float().cpu(), fixed_Gz.float().cpu(), intermediates



def plot_channel_activation(intermediates, img_index, img, experiment_name, config, state_dict, save_root):
    """plot intermediate activation for each channel
    param: 
        - intermediates: {layer_index: numpy.array(N, C, H, W)}
        - img_index: int, between 0 and N
    """
    print("Document img examined : img_idx {}")
    target_img_name = '{}/{}/{}/img_{}_iter_{}_target_.jpg'.format(save_root,
                                                    experiment_name, state_dict['itr'], 
                                                    img_index, state_dict['itr'])
    torchvision.utils.save_image(img, target_img_name, normalize=True)

    for layer_index in intermediates:
        layer_activation = intermediates[layer_index][img_index].unsqueeze(1) # [C, 1, H, W]
        C, _, H, W = layer_activation.shape
        activatity_channels = layer_activation.reshape(C, -1).std(1)
        sorted_act_index = torch.argsort(activatity_channels, descending=True)
        sorted_layer_activation = layer_activation[sorted_act_index]
        activation_visual_name = '%s/%s/%s/img_%s_iter_%s_plot_inter_activation_layer_%s.jpg' % (save_root,
                                                    experiment_name, state_dict['itr'], img_index, state_dict['itr'], layer_index)
        print("Ploting (img_idx {}) for layer {} activation shape {}".format(img_index, layer_index, layer_activation.shape))
        torchvision.utils.save_image(sorted_layer_activation.float(), activation_visual_name,
                                 nrow=int(layer_activation.shape[0] ** 0.5), normalize=True, padding=1, pad_value=1)

        # sum of channels
        activation_sum_name = '%s/%s/%s/img_%s_sum_iter_%s_plot_inter_activation_layer_%s.jpg' % (save_root,
                                                    experiment_name, state_dict['itr'], img_index, state_dict['itr'], layer_index)
        sum_activation = layer_activation.sum(0)
        torchvision.utils.save_image(sum_activation.float(), activation_sum_name,
                                 nrow=int(layer_activation.shape[0] ** 0.5), normalize=True, padding=1, pad_value=0.5)

        # ONE channel activation
        mean_activation = intermediates[layer_index].mean(0).reshape(C, -1).std(1) 
        select_one_ch = torch.argsort(mean_activation, descending=True)[0]
        ## plot all the activation on that channel
        one_ch_name = '%s/%s/%s/iter_%s_onech_plot_inter_activation_layer_%s.jpg' % (save_root,
                                                    experiment_name, state_dict['itr'], state_dict['itr'], layer_index)
        torchvision.utils.save_image(intermediates[layer_index][:, select_one_ch, :, :].unsqueeze(1).float(), one_ch_name,
                                 nrow=int(intermediates[layer_index].shape[0] ** 0.5), normalize=True, padding=1, pad_value=0.5)

        

        







# The main training file. Config is a dictionary specifying the configuration
# of this training run.

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
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = model.Generator(**config).to(device)
    D = model.Discriminator(**config).to(device)
    E = model.ImgEncoder(**config).to(device)
    # E = model.Encoder(**config).to(device)

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

    print('Number of params in G: {} D: {} E: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D, E]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, D, E, state_dict,
                           config['weights_root'], experiment_name,
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None)

    # If parallel, parallelize the GD module
    if config['parallel']:
        GDE = nn.DataParallel(GDE)
        if config['cross_replica']:
            patch_replication_callback(GDE)

    # Prepare data; the Discriminator's batch size is all that needs to be passed
    # to the dataloader, as G doesn't require dataloading.
    # Note that at every loader iteration we pass in enough data to complete
    # a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                    * config['num_D_accumulations'])
    loaders, train_dataset = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                        'start_itr': state_dict['itr']})

    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                               device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                         config['n_classes'], device=device,
                                         fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()
    print("fixed_y original: {} {}".format(fixed_y.shape, fixed_y[:10]))
    ## TODO: change the sample method to sample x and y
    fixed_x, fixed_y_of_x = utils.prepare_x_y(G_batch_size, train_dataset, experiment_name, config)
    

    # Build image pool to prevent mode collapes
    if config['img_pool_size'] != 0:
        img_pool = ImagePool(config['img_pool_size'], train_dataset.num_class,\
                                    save_dir=os.path.join(config['imgbuffer_root'], experiment_name),
                                    resume_buffer=config['resume_buffer'])
    else:
        img_pool = None

    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, E, GDE,
                                                ema, state_dict, config, img_pool)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                   else G),
                               z_=z_, y_=y_, config=config)




    # print('Beginning training at epoch %f...' % (state_dict['itr'] * D_batch_size / len(train_dataset)))
    print("Beginning testing at Epoch {} (iteration {})".format(state_dict['epoch'], state_dict['itr']))

    if config['G_eval_mode']:
        print('Switchin G to eval mode...')
        G.eval()
        if config['ema']:
            G_ema.eval()
    fixed_x, fixed_Gz, intermediates = activation_extract(G, D, E, G_ema, fixed_x, fixed_y_of_x, z_, y_,
                                state_dict, config, experiment_name, save_weights=config['save_weights'])
    
    plot_channel_activation(intermediates, config['img_index'], fixed_Gz[config['img_index']], experiment_name, config, state_dict)

def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
