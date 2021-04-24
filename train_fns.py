''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import os
from itertools import repeat

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


import utils
import losses
from plot_inter_activation import plot_channel_activation

# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}
    return train


def GAN_training_function(G, D, E, GDE, ema, state_dict, config, img_pool):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        E.optim.zero_grad()

        # How many chunks to split x and y into?
        x = torch.split(x, config['batch_size'])
        y = torch.split(y, config['batch_size'])
        # print("inside fns", x)
        print("split - x {}".format(len(x)))
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)
            utils.toggle_grad(E, False)
        # print("inside train fns: config['num_D_steps']", config['num_D_steps'])
        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()
            # print("---------------------- counter {} ---------------".format(counter))
            # print("x[counter] {}; y[counter] {}".format(x[counter].shape, y[counter].shape))
            for accumulation_index in range(config['num_D_accumulations']):
                # Cornner case for the last batch
                if counter >= len(x):
                    break
                D_fake, D_real = GDE(x[counter], y[counter], config, state_dict['itr'], img_pool, train_G=False,
                                    split_D=config['split_D'])
                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss( \
                    D_fake, D_real, config['clip'])
                D_loss = (D_loss_real + D_loss_fake) / \
                    float(config['num_D_accumulations'])
                print("D_loss: {}; D_fake {}, D_real {}".format(D_loss.item(), D_loss_fake.item(), D_loss_real.item()))
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])
            
            # stop gradient for testing purpose
            if config['stop_gradient']:
                print("!!! D is not optimized since you turn on `stop_gradient`!!!!!!")
            else:
                D.optim.step()

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)
            utils.toggle_grad(E, True)

        # Zero G/E's gradients by default before training G, for safety
        G.optim.zero_grad()
        E.optim.zero_grad()
        # If accumulating gradients, loop multiple times
        counter = 0 # reset counter for data split
        for accumulation_index in range(config['num_G_accumulations']):
            if counter >= len(x):
                    break
            # print("---------------------- counter {} ---------------".format(counter))
            output = GDE(x[counter], y[counter], config, state_dict['itr'], img_pool, train_G=True, split_D=config['split_D'], return_G_z=True)
            D_fake = output[0]
            G_z = output[2]
            mu, log_var = output[3], output[4]
            if len(output) == 6:
                G_additional = output[5]
            # print("checkpoint==========================")
            G_loss = losses.generator_loss(
                D_fake) / float(config['num_G_accumulations'])
            VAE_recon_loss = losses.vae_recon_loss(G_z, x[counter])
            VAE_kld_loss = losses.vae_kld_loss(mu, log_var, config['clip'])
            GE_loss = G_loss + VAE_recon_loss * config['lambda_vae_recon'] + VAE_kld_loss * config['lambda_vae_kld']
                            # weights_TTs.mean() * config['lambda_spatial_transform_weights']
                            
            # log_loss_str = f"GE_loss {GE_loss.item()}; VAE_recon_loss {VAE_recon_loss.item()}; VAE_kld_loss {VAE_kld_loss.item()}; weights_TTs {weights_TTs.mean().item()}; "
            log_loss_str = f"GE_loss {GE_loss.item()}; VAE_recon_loss {VAE_recon_loss.item()}; VAE_kld_loss {VAE_kld_loss.item()} "

            # add G_additional loss
            if len(output) == 6:
                G_additional_loss = config['lambda_g_additional'] * G_additional.sum()
                GE_loss += G_additional_loss
                log_loss_str += f"G_additional {G_additional_loss.item()}"
            
            # print out loss
            print(log_loss_str)
            
            # optimization
            GE_loss.backward()
            counter += 1


        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            # Debug print to indicate we're using ortho reg in G
            print('using modified ortho reg in G')
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                        blacklist=[param for param in G.shared.parameters()])
        
        # stop gradient for testing purpose
        if config['stop_gradient']:
            print("!!! G and E is not optimized since you turn on `stop_gradient`!!!!!!")
        else:
            G.optim.step()
            E.optim.step()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
               'D_loss_real': float(D_loss_real.item()),
               'D_loss_fake': float(D_loss_fake.item()),
               'VAE_recon_loss': float(VAE_recon_loss.item()),
               'VAE_KLD_loss': float(VAE_recon_loss.item())}
        # Return G's loss and the components of D's loss.
        return out
    return train


''' This function takes in the model, saves the weights (multiple copies if
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''


def save_and_sample(G, D, E, G_ema, fixed_x, fixed_y, z_, y_,
                    state_dict, config, experiment_name, img_pool, save_weights):

    # sparsity mode
    if 'gradient_topk_mode_' in config['sparsity_mode']:
        sparsity_mode = ('gradient_topk', float(config['sparsity_mode'].split('_')[-1]))
    elif 'regularized_sparse_hw_topk_mode_' in config['sparsity_mode']:
        sparsity_mode = ('regularized_sparse_hw_topk', float(config['sparsity_mode'].split('_')[-1]))
    else:
        sparsity_mode = ("", 0.0)
    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G

    if save_weights:
        utils.save_weights(G, D, E, state_dict, config['weights_root'],
                           experiment_name, None, G_ema if config['ema'] else None)
        # Save an additional copy to mitigate accidental corruption if process
        # is killed during a save (it's happened to me before -.-)
        if config['num_save_copies'] > 0:
            utils.save_weights(G, D, E, state_dict, config['weights_root'],
                               experiment_name,
                               'copy%d' % state_dict['save_num'],
                               G_ema if config['ema'] else None)
            state_dict['save_num'] = (
                state_dict['save_num'] + 1) % config['num_save_copies']



        # Accumulate standing statistics?
        if config['accumulate_stats']:
            utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                            z_, y_, config['n_classes'],
                                            config['num_standing_accumulations'])

    # Save a random sample sheet with fixed z and y
    # TODO: change here to encode fixed x into z and feed z with fixed y into G
    with torch.no_grad():
        # if config['parallel']:
        #     print("fixed_x: ", fixed_x.shape)
        #     if config['inference_nosample']:
        #         _, fixed_z, _ = [i.detach() for i in nn.parallel.data_parallel(E, fixed_x)]
        #     else:
        #         fixed_z, _, _ =  [i.detach() for i in nn.parallel.data_parallel(E, fixed_x)]
        #     print("fixed_z: ", fixed_z.shape)
        #     print("fixed_y: ", fixed_y.shape)

        #     output = nn.parallel.data_parallel(
        #             which_G, (fixed_z, which_G.shared(fixed_y), state_dict['itr'], repeat(None),\
        #                      repeat(True)))
            
        #     fixed_Gz = output[0].detach()
        #     intermediates = output[1]
        #     # else:
        #     #     mask_x_all = [item.detach() for item in output[1]] # [[n, k, h, w],...,]

        # else:

        fixed_z, _, _ = E(fixed_x)
        print("fixed_x: ", fixed_x.shape)
        print("fixed_z: ", fixed_z.shape)
        print("fixed_y: ", fixed_y.shape)
        output = which_G(fixed_z, which_G.shared(fixed_y), state_dict['itr'], return_inter_activation=True)
        
        fixed_Gz = output[0].detach()
        intermediates = output[1]
    
    

    
    n, c, h, w = fixed_Gz.shape
    # log masks 
    save_dir = '%s/%s/%s' % (config['samples_root'], experiment_name, state_dict['itr'])
    os.makedirs(save_dir, exist_ok=True)
    image_filename = save_dir + '/fixed_samples%d.jpg' % (state_dict['itr'])

    torchvision.utils.save_image(fixed_Gz, image_filename,
                                 nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)

    print(f"sparsity_mode {sparsity_mode}")
    if sparsity_mode[0] in ['gradient_topk', 'regularized_sparse_hw_topk']: 
        
        # save each channel
        image_indexs = config['eval_image_indexs']
        block_names = intermediates['sparse_block']
        print(f"block_names {block_names}")
        for block in block_names:
            activation = intermediates[block].cpu()
            # print(f"activation {activation.shape}")
        
            for image_index in image_indexs:
                # print(f"image_index {image_index}")
                channel_image_filename = '%s/eval_iter_%s_img_%s_block_%s_channels_.jpg' % (save_dir,
                                                                state_dict['itr'],
                                                                str(image_index), 
                                                                block)
                
                # sort activation by activation magnitude
                # activation_sort = torch.argsort(activation[image_index].std((1,2)), descending=True)
                activation_i = activation[image_index].unsqueeze(1).repeat(1, 3, 1, 1)
                
                torchvision.utils.save_image(activation_i.float(), channel_image_filename,
                                            nrow=int(activation_i.shape[0] ** 0.5), normalize=True)

                # activation 10%?
                sparsity = torch.where(activation_i == 0)[0]
                print(f"Sparsity: {1 - sparsity.shape[0] / (activation_i.shape[0] * 3 * activation_i.shape[2] * activation_i.shape[3])}")


    

    # if True and mask_x_all:
    #     for layers in range(len(mask_x_all)):
    #         layer_map_name = os.path.join(save_dir, f"layer_{layers}_mask.jpg")
    #         mask_i = mask_x_all[layers].float().cpu() # [n, k, h_m, w_m]
    #         n, k, h, w = mask_i.shape
    #         # mask_i dot product
    #         mask_i_TT = torch.matmul(mask_i.view(n, k, -1), torch.transpose(mask_i.view(n, k, -1), 1, 2)) # n, k, k
    #         mask_i_TT = mask_i_TT.unsqueeze(1).repeat(1, 3, 1, 1) # n, 3, k, k
    #         cov_name = os.path.join(save_dir, f"layer_{layers}_mask_cov.jpg")
    #         torchvision.utils.save_image(mask_i_TT, cov_name, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
    #         print(f"saved {cov_name}...")
    #         # # upsample latent map to real map
    #         # assert h % h_m == 0, f"mask {h_m}x{h_m} is not a factor of h {h}x{h}"
    #         # scale_factor = h // h_m
    #         # mask_i = F.interpolate(mask_i, scale_factor=scale_factor) # n, k, h, w

    #         mask_sum = mask_i.sum(1, keepdim=True).repeat(1, 3, 1, 1) # n, 3, h, w
            
    #         mask_i = mask_i.view(n, k, 1, h, w).repeat(1, 1, 3, 1, 1) # n, k, 3, h, w

            
            
    #         # print(f"fixed_Gz {fixed_Gz.unsqueeze(1).shape}")
    #         print(f"mask_sum { mask_sum.unsqueeze(1).shape}")
    #         print(f"mask_i {mask_i.shape}")
    #         # mask_img = torch.cat([fixed_Gz.float().cpu().unsqueeze(1), mask_sum.unsqueeze(1), mask_i], dim=1) # n, k+2, 3, h, w
    #         mask_img = torch.cat([mask_sum.unsqueeze(1), mask_i], dim=1) # n, k+2, 3, h, w
    #         torchvision.utils.save_image(mask_img.view(-1, 3, h, w), layer_map_name, nrow=k+1, normalize=True)
            
            # TODO: why the mask looks all the same for every images? Is the latent code all the same? for different n? should you use vae? 



    
    # if (not config['no_adaptive_tau']) and (state_dict['itr'] * config['sparse_decay_rate'] < 1.1):
    #     image_filename_train = '%s/%s/fixed_samples%d_train.jpg' % (config['samples_root'],
    #                                                 experiment_name,
    #                                                 state_dict['itr'])

    #     torchvision.utils.save_image(fixed_Gz_train.float().cpu(), image_filename_train,
    #                              nrow=int(fixed_Gz_train.shape[0] ** 0.5), normalize=True)
    # # For now, every time we save, also save sample sheets
    # utils.sample_sheet(which_G,
    #                    classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
    #                    num_classes=config['n_classes'],
    #                    samples_per_class=10, parallel=config['parallel'],
    #                    samples_root=config['samples_root'],
    #                    experiment_name=experiment_name,
    #                    folder_number=state_dict['itr'],
    #                    z_=z_,
    #                    iter_num=int(1 / config['sparse_decay_rate']) + 100)
    # # Also save interp sheets
    # for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    #     utils.interp_sheet(which_G,
    #                        num_per_sheet=16,
    #                        num_midpoints=8,
    #                        num_classes=config['n_classes'],
    #                        parallel=config['parallel'],
    #                        samples_root=config['samples_root'],
    #                        experiment_name=experiment_name,
    #                        folder_number=state_dict['itr'],
    #                        sheet_number=0,
    #                        fix_z=fix_z, fix_y=fix_y, device='cuda',
    #                        iter_num=int(1 / config['sparse_decay_rate']) + 100)

    # plot the intermediate layer activation
    # plot_channel_activation(intermed_activation, config['img_index'], fixed_Gz[config['img_index']], experiment_name, config, state_dict, config['samples_root'])
    
    # Save ImagePool
    if config['img_pool_size'] != 0:
        img_pool.save()




''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID,
    user-specified), logs the results, and saves a best_ copy if it's an
    improvement. '''


def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                        z_, y_, config['n_classes'],
                                        config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,
                                                 config['num_inception_images'],
                                                 num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' %
          (state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
        print('%s improved over previous best, saving checkpoint...' %
              config['which_best'])
        utils.save_weights(G, D, state_dict, config['weights_root'],
                           experiment_name, 'best%d' % state_dict['save_best_num'],
                           G_ema if config['ema'] else None)
        state_dict['save_best_num'] = (
            state_dict['save_best_num'] + 1) % config['num_best_copies']
    state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
    state_dict['best_FID'] = min(state_dict['best_FID'], FID)
    # Log results to file
    test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                 IS_std=float(IS_std), FID=float(FID))
