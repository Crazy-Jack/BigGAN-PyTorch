import numpy as np
import math
import functools
from itertools import repeat

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision.models as vision_models
from torchsummary import summary


import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111', sparsity_resolution='8_16_32_64', sparsity_ratio='20_10_10_5', no_sparsity=False):
    arch = {}
    if not no_sparsity:
        assert len(sparsity_resolution.split('_')) == len(sparsity_ratio.split('_')), "length sparsity and sparsity_ratio doesn't match"
        sparsity_pairs = {pair[0]: 0.01 * pair[1] for pair in zip([int(resolute) for resolute in sparsity_resolution.split('_')], [int(ratio) for ratio in sparsity_ratio.split('_')])}
    else:
        sparsity_pairs = {}
    print("G_arch sparsity_pairs", sparsity_pairs)
    arch[512] = {'in_channels':  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
                 'upsample': [True] * 7,
                 'resolution': [8, 16, 32, 64, 128, 256, 512],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                               for i in range(3, 10)},
                 'sparsity':  {**{i:False for i in [8, 16, 32, 64, 128, 256, 512] if i not in sparsity_pairs}, **sparsity_pairs}}
    arch[256] = {'in_channels':  [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16,  8, 8, 4, 2, 1]],
                 'upsample': [True] * 6,
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                               for i in range(3, 9)},
                 'sparsity':  {**{i:False for i in [8, 16, 32, 64, 128] if i not in sparsity_pairs}, **sparsity_pairs}}
    arch[128] = {'in_channels':  [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [True] * 5,
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)},
                 'sparsity':  {**{i:False for i in [8, 16, 32, 64, 128] if i not in sparsity_pairs}, **sparsity_pairs}}
    arch[64] = {'in_channels':  [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)},
                'sparsity':  {**{i:False for i in [8, 16, 32, 64] if i not in sparsity_pairs}, **sparsity_pairs}}
    arch[32] = {'in_channels':  [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)},
                'sparsity':  {**{i:False for i in [8, 16, 32] if i not in sparsity_pairs}, **sparsity_pairs}}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn', sparsity_resolution='', sparsity_ratio='', no_sparsity=True, mask_base=1e-2,
                 sparsity_mode="spread", sparse_decay_rate=1e-4, no_adaptive_tau=False, local_reduce_factor=4, test_layer=-1, 
                 test_target_block="", select_index=-1, gumbel_temperature=1.0, 
                 conv_select_kernel_size=5, vc_dict_size=150, sparse_vc_interaction_num=4, sparse_vc_prob_interaction=4, 
                 test_all=False, **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Sparsity 
        self.sparsity_resolution, self.sparsity_ratio = sparsity_resolution, sparsity_ratio
        self.no_sparsity = no_sparsity
        self.sparsity_mode = sparsity_mode
        self.sparse_decay_rate = sparse_decay_rate
        self.no_adaptive_tau = no_adaptive_tau
        self.local_reduce_factor = local_reduce_factor
        self.test_layer = None if test_layer == -1 else test_layer
        self.test_target_block = None if test_target_block == "" else [int(i) for i in test_target_block.split("_")]
        self.select_index = [int(i) for i in select_index.split("_")] if "_" in select_index else int(select_index) # for eval single vc
        self.gumbel_temperature = gumbel_temperature

        # conv select
        self.conv_select_kernel_size = conv_select_kernel_size
        self.vc_dict_size = vc_dict_size
        self.sparse_vc_interaction_num = sparse_vc_interaction_num
        self.sparse_vc_prob_interaction = sparse_vc_prob_interaction
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention, sparsity_resolution=self.sparsity_resolution, \
                                sparsity_ratio=self.sparsity_ratio, no_sparsity=self.no_sparsity)[resolution]
        print("G arch sparsity: ", self.arch['sparsity'])
        self.mask_base = mask_base
        self.test_all = test_all

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(
                nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        self.which_bn = functools.partial(layers.ccbn,
                                          which_linear=bn_linear,
                                          cross_replica=self.cross_replica,
                                          mybn=self.mybn,
                                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                                      else self.n_classes),
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                       else layers.identity())
        # First linear layer
        self.linear = self.which_linear(self.dim_z // self.num_slots,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           which_bn=self.which_bn,
                                           activation=self.activation,
                                           upsample=(functools.partial(F.interpolate, scale_factor=2)
                                                     if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' %
                      self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(
                    self.arch['out_channels'][index], self.which_conv)]
            
            if not no_sparsity:
                # If sparsity on this block, attach it to the end
                sparse_percent = self.arch['sparsity'][self.arch['resolution'][index]]
                if sparse_percent:
                    print('Adding sparsity layer in G at resolution %d' %
                        self.arch['resolution'][index])
                    if self.sparsity_mode == "direct":
                        self.blocks[-1] += [layers.Sparsify_all(sparse_percent, self.mask_base)]
                    elif self.sparsity_mode == "spread":
                        print("### Adding sparsity in a spread manner...")
                        sparse_percent = np.sqrt(sparse_percent).item()
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent), layers.Sparsify_hw(sparse_percent)]
                    elif self.sparsity_mode[:5] == "hyper": # for any hyper mode of sparsity
                        print(f"### Adding sparsity in a {sparsity_mode} manner...")
                        # sparse_percent = np.sqrt(sparse_percent).item()
                        # self.blocks[-1] += [layers.Sparsify_ch(sparse_percent), layers.Sparsify_hypercol(sparse_percent, mode=sparsity_mode)]
                        self.blocks[-1] += [layers.Sparsify_hypercol(sparse_percent, mode=sparsity_mode, \
                                                ch=self.arch['out_channels'][index], which_conv=self.which_conv, \
                                                resolution=self.arch['resolution'][index])]
                        # adding attention layer if perform hypercolum selection
                        print('Adding attention layer in G at resolution %d after hypercol sparsity...  ' %
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Attention(
                            self.arch['out_channels'][index], self.which_conv)]

                    elif self.sparsity_mode == 'local_modular_hyper_col':  # topk, ch, which_conv, resolution
                        print('Adding local modular hypercolumn selection layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])

                        self.blocks[-1] += [layers.Sparsify_hypercol_local_modular(sparse_percent, \
                                                self.arch['out_channels'][index], which_conv=self.which_conv, \
                                                resolution=self.arch['resolution'][index], local_reduce_factor=self.local_reduce_factor)]
                    elif self.sparsity_mode == 'combine_vc_sparse_bottleneck':
                        print('Adding combine_vc_sparse_bottleneck layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparse_vc_combination(sparse_percent, self.which_conv, self.arch['out_channels'][index], \
                                                                            self.arch['resolution'][index], gumbel_temperature=self.gumbel_temperature)]
                    elif self.sparsity_mode == 'vc_map_combination':
                        print('Adding vc_map_combination layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparse_vc_map_combination(sparse_percent, self.which_conv, self.arch['out_channels'][index], \
                                                                                self.arch['resolution'][index])]
                    elif self.sparsity_mode == 'implicit_sparse_vc_recover':
                        print('Adding implicit_sparse_vc_recover layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent), 
                                            layers.Implicit_sparse_vc_recover(sparse_percent, self.arch['out_channels'][index], \
                                                                                self.arch['resolution'][index])]
                    elif self.sparsity_mode == 'implicit_nonsparse_vc_recover':
                        print('Adding implicit_nonsparse_vc_recover layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Implicit_sparse_vc_recover(sparse_percent, self.arch['out_channels'][index], \
                                                                                self.arch['resolution'][index])]
                    elif self.sparsity_mode == 'implicit_sparse_vc_recover_vcsparse':
                        print('Adding implicit_sparse_vc_recover_vcsparse layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent), 
                                            layers.Implicit_sparse_vc_recover(sparse_percent, self.arch['out_channels'][index], \
                                                                            self.arch['resolution'][index], vc_go_sparse=True)]
                    elif self.sparsity_mode == 'implicit_sparse_vc_recover_vcsparse_conditional':
                        print('Adding implicit_sparse_vc_recover_vcsparse_conditional layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent), 
                                            layers.Implicit_sparse_vc_recover(sparse_percent, self.arch['out_channels'][index], \
                                                                            self.arch['resolution'][index], vc_go_sparse=True, \
                                                                            y_share_dim=self.shared_dim)]

                    elif self.sparsity_mode == 'implicit_sparse_vc_recover_vcsparse_weight_sp':
                        print('Adding implicit_sparse_vc_recover_vcsparse_weight_sp layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent),
                                            layers.Implicit_sparse_vc_recover(sparse_percent, self.arch['out_channels'][index], \
                                                                            self.arch['resolution'][index], vc_go_sparse=True, \
                                                                            spatial_implicit_comb_sparse_weight=True)]
                    elif self.sparsity_mode == 'implicit_sparse_vc_recover_vcsparse_weight_sp_maskoutput':
                        print('Adding implicit_sparse_vc_recover_vcsparse_weight_sp layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent),
                                            layers.Implicit_sparse_vc_recover(sparse_percent, self.arch['out_channels'][index], \
                                                                            self.arch['resolution'][index], vc_go_sparse=True, \
                                                                            spatial_implicit_comb_sparse_weight=True, 
                                                                            mask_reconstruct=True)]
                    elif self.sparsity_mode == 'conv_sparse_vc_recover':
                        print('Adding conv_sparse_vc_recover layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.conv_sparse_mode = "1.0"
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent),
                                            layers.SparseNeuralConv(sparse_percent, self.arch['out_channels'][index], \
                                            self.arch['resolution'][index], kernel_size=self.conv_select_kernel_size, \
                                            vc_dict_size=self.vc_dict_size)]
                    elif self.sparsity_mode == 'conv_sparse_vc_recover_no_vcattention':
                        print('Adding conv_sparse_vc_recover_no_vcattention layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.conv_sparse_mode = "1.0"
                        self.blocks[-1] += [layers.Sparsify_ch(sparse_percent),
                                            layers.SparseNeuralConv(sparse_percent, self.arch['out_channels'][index], \
                                            self.arch['resolution'][index], kernel_size=self.conv_select_kernel_size, \
                                            vc_dict_size=self.vc_dict_size, no_attention=True)]
                    elif self.sparsity_mode == 'conv_sparse_vc_recover_no_sparse':
                        print('Adding conv_sparse_vc_recover_no_sparse layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.conv_sparse_mode = "1.0"
                        self.blocks[-1] += [layers.SparseNeuralConv(sparse_percent, self.arch['out_channels'][index], \
                                            self.arch['resolution'][index], kernel_size=self.conv_select_kernel_size, \
                                            vc_dict_size=self.vc_dict_size)]
                    elif self.sparsity_mode == 'conv_sparse_vc_recover_no_sparse_noattention':
                        print('Adding conv_sparse_vc_recover_no_sparse_noattention layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.conv_sparse_mode = "1.0"
                        self.blocks[-1] += [layers.SparseNeuralConv(sparse_percent, self.arch['out_channels'][index], \
                                            self.arch['resolution'][index], kernel_size=self.conv_select_kernel_size, \
                                            vc_dict_size=self.vc_dict_size, no_attention_select=True)]

                    elif 'conv_sparse_vc_recover_no_sparse_mode_' in self.sparsity_mode:
                        print(f'Adding {self.sparsity_mode} layer in G at resolution %d ... ' % 
                            self.arch['resolution'][index])
                        self.conv_sparse_mode = self.sparsity_mode.split("_")[-1]
                        self.blocks[-1] += [layers.SparseNeuralConv(sparse_percent, self.arch['out_channels'][index], \
                                            self.arch['resolution'][index], kernel_size=self.conv_select_kernel_size, \
                                            vc_dict_size=self.vc_dict_size, 
                                            mode=self.conv_sparse_mode,
                                            sparse_vc_interaction=self.sparse_vc_interaction_num, 
                                            sparse_vc_prob_interaction=self.sparse_vc_prob_interaction,
                                            test=self.test_all)]
                    
                    else:
                        raise NotImplementedError(f"Sparsity Mode Invalid: {self.sparsity_mode}")

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block)
                                     for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0,
                                      eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement()
                                         for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' %
              self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y, iter_num, y_origin=None, return_inter_activation=False, device='cuda', normal_eval=True, eval_vc=False, return_mask=False):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0] 
            # print("INside G: forward y {}; first z {} ".format(y.shape, z.shape))
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
            
        else:
            ys = [y] * len(self.blocks)

        if self.sparsity_mode in ['implicit_sparse_vc_recover_vcsparse_conditional']:
            ys_self = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z) 
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # prepare list for intermediate output
        intermediates = {}
        # spatial transform weight regularization
        weight_TTs = 0
        mask_x_all = []
        prob_vects = []
        previous_prob_vects = []
        affinity_map = []
        entropys = 0

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block_idx, block in enumerate(blocklist):
                # print("block: h {} ; ys[index] {}".format(h.shape, ys[index].shape))
                if block.myid in ['sparse_ch', 'sparse_hw', 'sparse_all', 'sparse_hyper']:
                    # decay linearly:
                    if self.no_adaptive_tau:
                        tau = 1.0
                    else:
                        tau = min(iter_num * self.sparse_decay_rate, 1)
                    h = block(h, tau, device=device)
                elif block.myid == 'local_modular_hyper_col':
                    if (not self.training) and (self.test_layer == index) and (not normal_eval):
                        print("inside bigGAN, self.test_target_block", self.test_target_block)
                        test_top1_blockindex = self.test_target_block # this block is spatial block in local_modular_hyper_col, not block_idx
                    else:
                        print("layer {} is active and not effected ".format(index))
                        test_top1_blockindex = None # the normal version

                    h = block(h, test_top1_blockindex, self.select_index, device=device)
                elif block.myid == 'vc_map_combination':
                    if self.no_adaptive_tau:
                        tau = 0.1
                    else:
                        tau = max(1 - iter_num * self.sparse_decay_rate, 0.1)

                    h = block(h, temp=tau)
                elif block.myid == 'implicit_sparse_vc_recover':
                    if self.sparsity_mode in ['implicit_sparse_vc_recover_vcsparse_conditional']:
                        class_info = ys_self[index]
                    else:
                        class_info = None
                        # print(f"class info {class_info}")

                    if eval_vc and (self.test_layer == index) and (not normal_eval):
                        print(f"testing layer {index} ")
                        h, weight_TT, mask_x = block(h, device=device, class_info=class_info, eval_vc_index=self.select_index)
                    else:
                        h, weight_TT, mask_x = block(h, device=device, class_info=class_info)
                    
                    weight_TTs += weight_TT
                    mask_x_all.append(mask_x)
                elif block.myid == "conv_sparse_vc_recover":
                    if eval_vc and (self.test_layer == index) and (not normal_eval):
                        print(f"testing layer {index} ")
                        h, (mask_x, prob_vector, previous_prob_vector, origin_map) = block(h, eval_=True, select_index=self.select_index)
                        
                    else:
                        if self.conv_sparse_mode == '1.0':
                            h, (mask_x, prob_vector, previous_prob_vector, origin_map) = block(h)
                            mask_x_all.append(mask_x)
                            prob_vects.append(prob_vector)
                            previous_prob_vects.append(previous_prob_vector)
                            affinity_map.append(origin_map)
                        elif float(self.conv_sparse_mode) >= 2.0 and float(self.conv_sparse_mode) < 5.0:
                            h, _ = block(h)
                        elif float(self.conv_sparse_mode) >= 5.0:
                            h, entropy = block(h)
                            entropys += entropy
                else:
                    # one hack
                    if self.test_all:
                        print(f"##################################################### ys[index] {type(ys[index])}")
                        print(f"##################################################### ys[index] {ys[index].shape}")
                        target_num = h.shape[0]
                        y_hack = torch.gather(ys[index], dim=0, index=torch.zeros((target_num, ys[index].shape[1])).long().cuda())
                        print(f"y_hack {y_hack.shape}")
                        print(f"block id {block.myid}")
                        if block.myid in ['atten']:
                            h = block(h, y_hack)
                        else:
                            h = block(h, y_hack, nobn=False)
                        
                    else:
                        h = block(h, ys[index])

                    
                # impose sparsity constraint for the activation
                # if index == 0:
                #     h = layers.sparsify_layer(h, sparsity=sparsity, device=device)
                
                if return_inter_activation:
                    out = h.detach()
                    intermediates["{}-{}".format(index, block_idx)] = out
                    print("Get activation from block {}-{} : {} ----------- ".format(index, block_idx, out.shape))
        
        # output dict
        
        # Apply batchnorm-relu-conv-tanh at output
        if return_inter_activation:
            return torch.tanh(self.output_layer(h)), intermediates
        
        if return_mask:
            return torch.tanh(self.output_layer(h)), (mask_x_all, prob_vects, previous_prob_vects, affinity_map)
        # adding maximum entropy to encourage exploration
        if entropys != 0:
            return torch.tanh(self.output_layer(h)), entropys
        
        return torch.tanh(self.output_layer(h)), None


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels':  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels':  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels':  [3] + [ch*item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels':  [3] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', patchGAN=False, **kwargs):
        super(Discriminator, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]
        self.patchGAN = patchGAN # bool indicates whether use patchGAN or not

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' %
                      self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block)
                                     for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        if not self.patchGAN:
            # normal activation
            self.linear = self.which_linear(
                self.arch['out_channels'][-1], output_dim)
            # Embedding for projection discrimination
        else:
            # patch GAN, impose a conv layer to 1 channel with tanh activation
            self.patch_conv = self.which_conv(self.arch['out_channels'][-1], 1)
            # TODO: add class conditional for patch GAN
            

        self.embed = self.which_embedding(
            self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement()
                                         for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' %
              self.param_count)

    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        if not self.patchGAN:
            # Apply global sum pooling as in SN-GAN
            h = torch.mean(self.activation(h), [2, 3])
            # Get initial class-unconditional output
            out = self.linear(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out = out + torch.mean(self.embed(y) * h, 1, keepdim=True)
        else:
            # print(f"Output patch size before {h.shape}")
            # Return a patch of activation 
            h = self.patch_conv(self.activation(h)) # [n, 1, h, w]
            # print(f"Output patch size {h.shape}")
            out = h 
        return out

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.


class G_D_E(nn.Module):
    def __init__(self, G, D, E):
        super(G_D_E, self).__init__()
        self.G = G
        self.D = D
        self.E = E 

    def forward(self, x, y, config, iter_num, img_pool, train_G=False, return_G_z=False,
                split_D=False, verbose=False):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Encode image by VAE
            z, mean, logvar = self.E(x)
            # print("z", z[0])
            # print("mean", mean[0])
            # print("logvar", logvar[0])
            # Get Generator output given noise
            if verbose:
                print("x inside GDE", x.shape)
                print("y", y.shape)
                print("self.G.shared(y)", self.G.shared(y).shape)
                print("z", z.shape)
            output = self.G(z, self.G.shared(y), iter_num, y_origin=y)
            G_z = output[0]
            G_additional = None
            if output[1] and (len(output)==2):
                G_additional = output[1]
            if not config['no_sparsity']:
                pass
                # weight_TTs = output[1]
                # mask_x_all = output[2]
                weight_TTs, mask_x_all = None, None

            if not train_G:
                if img_pool:
                    G_z = img_pool.query(G_z, y) # when train discriminator, use buffered generated image to avoid mode collapes, not when train_G
            # print("G_z", G_z[0])
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        if split_D:
            D_fake = self.D(G_z, y)
            D_real = self.D(x, y)
            return D_fake, D_real

        # If real data is provided, concatenate it with the Generator's output
        # along the batch dimension for improved efficiency.
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([y, y], 0) if y is not None else y
            # Get Discriminator output
            D_out = self.D(D_input, D_class)
            if not return_G_z:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])
            else:
                D_fake, D_real = torch.split(D_out, [G_z.shape[0], x.shape[0]])
                if config['no_sparsity']:
                    return D_fake, D_real, G_z, mean, logvar
                elif G_additional:
                    return D_fake, D_real, G_z, mean, logvar, G_additional
                else:
                    return D_fake, D_real, G_z, mean, logvar, weight_TTs, mask_x_all




##########################################
#            VAE Encoder                 #
##########################################
class LayerNorm2D(torch.nn.Module):
    def __init__(self, num_features, eps=1e-6, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1))
    def forward(self, x):
        assert x.dim() == 4
        std_new, mean_new = torch.std_mean(x, dim=(1,2,3),keepdim=True)
        x_new = (x-mean_new)/(std_new + self.eps)
        if self.affine:
            return x_new * self.gamma + self.beta
        return x_new

def convert_bn(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_feat = child.num_features
            setattr(model, child_name, LayerNorm2D(num_feat, affine=True))
        else:
            convert_bn(child)

def convert_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            convert_relu(child)

class ImgEncoder(nn.Module):
    """Encoder for VAE"""
    def __init__(self, dim_z, shared_dim, E_lr, E_B1, E_B2, adam_eps=1e-8, in_shape=3, encoder='Resnet-18', **kwargs):
        super(ImgEncoder, self).__init__()
        if encoder == 'Resnet-50':
            orig_network = vision_models.resnet50(pretrained=False)
            self.outdim = 2048
        elif encoder == 'Resnet-18':
            orig_network = vision_models.resnet18(pretrained=False)
            self.outdim = 512
        elif encoder == 'Resnet-34':
            orig_network = vision_models.resnet34(pretrained=False)
            self.outdim = 512
        else:
            raise NotImplementedError("encoder type {} not supported.".format(encoder))
        
        # customize for resnet
        if encoder in ['Resnet-50', 'Resnet-34', 'Resnet-18']:
            convert_bn(orig_network)
            convert_relu(orig_network)
            if in_shape!=3:
                orig_network._modules['conv1'] = nn.Conv2d(in_channels=in_shape, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.features = nn.Sequential(*list(orig_network.children())[:9])
            
            # self.features = nn.Sequential(*list(orig_network.children()))
            # summary(self.features, (3, 128, 128))
        else:
            raise NotImplementedError("encoder type {} not supported.".format(encoder))

        if shared_dim > 0:
            dim_z = shared_dim
        self.mean_head = torch.nn.Linear(self.outdim, dim_z)
        self.std_head = torch.nn.Linear(self.outdim, dim_z)

        # set optimizer
        self.lr, self.B1, self.B2, self.adam_eps = E_lr, E_B1, E_B2, adam_eps
        self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    def forward(self, x):
        x_out = self.features(x)
        # print("x_out Encoder {}; input x Encoder: {}".format(x_out.shape, x.shape))
        x_out = x_out.view(-1, self.outdim)
        # print("x_out Encoder {}; input x Encoder: {}".format(x_out.shape, x.shape))
        mean_, logvar_ = self.mean_head(x_out), self.std_head(x_out)
        z_sample = self.reparameter(mean_, logvar_)
        return z_sample, mean_, logvar_ 
    
    def reparameter(self, mean, logvar):
        """reparameter and return a sampled feature veector"""
        # reparameterization trick for VAE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mean
        return z


#################################################
#        VAE ENCODER based on Discriminator     #
#################################################
class Encoder(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 E_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, dim_z=120, D_mixed_precision=False, D_fp16=False,
                 E_init='ortho', skip_init=False, D_param='SN', **kwargs):
        super(Encoder, self).__init__()
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = E_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in E at resolution %d' %
                      self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block)
                                     for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.mean_linear = self.which_linear(
            self.arch['out_channels'][-1], dim_z)
        self.logvar_linear = self.which_linear(
            self.arch['out_channels'][-1], dim_z)

        # Initialize weights
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = E_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in E...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement()
                                         for p in module.parameters()])
        print('Param count for E''s initialized parameters: %d' %
              self.param_count)

    def forward(self, x):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        self.mean = self.mean_linear(h)
        self.logvar = self.logvar_linear(h)
        # reparameterized trick
        # print("mean before repraram", self.mean)
        z_sample = self.reparameter(self.mean, self.logvar) # (N, latent_dim)
        return z_sample, self.mean, self.logvar
    
    def reparameter(self, mean, logvar):
        """reparameter and return a sampled feature veector"""
        # reparameterization trick for VAE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mean
        return z