''' Layers
    This file contains various layers for the BigGAN models.
'''
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable


from layer_conv_select_generate_mask import *
from layer_recon import *
#################################################
#         idea: convolutional selection         #
#################################################
"""perform hypercolumn sparsity since within each scope"""

class NeuralConvSelection(nn.Module):
    """conv select"""
    def __init__(self, ch, resolution, kernel_size, vc_dict_size=100, no_attention=False, hard_selection=False, which_mask="1.0", \
                    sparse_vc_prob_interaction=4, vc_type="parts", test=False, pull_vc_activation=0.25):
        super(NeuralConvSelection, self).__init__()
        self.kernel_size = kernel_size
        self.which_mask = which_mask
        self.test = test 
        self.pull_vc_activation = pull_vc_activation
        if which_mask == '1.0':
            self.generate_mask = GenerateMask(ch, resolution=kernel_size, or_cadidate=vc_dict_size, no_attention=no_attention) # shared generate mask module, generate a weighted mask for 
        elif float(which_mask) >= 2.0 and (float(which_mask) < 3.0):
            self.generate_mask = GenerateMask_2_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size) # shared generate mask module, generate a weighted mask for 
            self.integrate_mask_activation = nn.Sequential(
                nn.Linear(ch + kernel_size * kernel_size, int(ch * 3)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(ch * 3), ch),
                nn.Tanh()
            )
        elif (float(which_mask) >= 3.0) and (float(which_mask) < 5.0):
            self.generate_mask = GenerateMask_3_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size, sparse_vc_prob_interaction=sparse_vc_prob_interaction, vc_type=vc_type) # shared generate mask module, generate a weighted mask for 
            self.integrate_mask_activation = nn.Sequential(
                nn.Linear(ch + kernel_size * kernel_size, int(ch * 3)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(ch * 3), ch),
                nn.Tanh()
            )
        elif (float(which_mask) >= 5.0) and (float(which_mask) < 6.0):
            print("=== in select build 5.0 mask ===")
            self.generate_mask = GenerateMask_3_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size, sparse_vc_prob_interaction=sparse_vc_prob_interaction, 
                                                    vc_type=vc_type, reg_entropy=True) # regularize the negative entropy of vc prob
            self.integrate_mask_activation = nn.Sequential(
                nn.Linear(ch + kernel_size * kernel_size, int(ch * 3)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(ch * 3), ch),
                nn.Tanh()
            )
        elif float(which_mask) >= 6.0 and (float(which_mask) < 7.0):
            if float(which_mask) == 6.1:
                self.generate_mask = GenerateMask_3_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size, \
                                                    sparse_vc_prob_interaction=sparse_vc_prob_interaction, \
                                                    warmup=54360, 
                                                    vc_type=vc_type, reg_entropy=True, no_map=True, pull_vc_activation=[self.pull_vc_activation,])
            elif float(which_mask) == 6.2:
                self.generate_mask = GenerateMask_3_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size, \
                                                    sparse_vc_prob_interaction=sparse_vc_prob_interaction, \
                                                    warmup=54360, 
                                                    vc_type=vc_type, reg_entropy=True, no_map=True, \
                                                    pull_vc_activation=[self.pull_vc_activation,], replace_activation=True)
            else:
                self.generate_mask = GenerateMask_3_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size, sparse_vc_prob_interaction=sparse_vc_prob_interaction, 
                                                    vc_type=vc_type, reg_entropy=True, no_map=True) # regularize the negative entropy of vc prob
        elif float(which_mask) >= 7.0 and (float(which_mask) < 8.0):
            # condition on vc to force network learn different output
            if float(which_mask) == 7.0:
                self.generate_mask = GenerateMask_3_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size, \
                                                    sparse_vc_prob_interaction=sparse_vc_prob_interaction, \
                                                    warmup=54360,
                                                    vc_type=vc_type, reg_entropy=True, no_map=True, pull_vc_activation=[self.pull_vc_activation,])
            
        else:
            raise NotImplementedError(f"which_mask {which_mask} is not implemented in NeuralConvSelection module")
        
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), stride=3)

    def forward(self, x):
        """
        x: [n, c, h, w]
        output: [n, L, c]
        """
        n, c, h, w = x.shape
        x = self.unfold(x) # [n, c * kernel * kernel, L]
        L = x.shape[2]
        x = torch.transpose(x, 1, 2).view(n, -1, c, self.kernel_size, self.kernel_size) # [n, L, c, kernel, kernel]
        # print(f"x shape before mask {x.shape}")
        if self.which_mask == '1.0':
            x_mask, prob_vector, prob_vector_previous, origin_dot_product = self.generate_mask(x) # [n * L, 1, kernel, kernel]
        
            x = (x_mask * x.reshape(-1, c, self.kernel_size, self.kernel_size)).sum((2,3)) # [n * L, c, kernel, kernel] -> [n * L, c]
            x = x.view(-1, L, c) # [n, L, c]
            return x, (x_mask.reshape(n, L, self.kernel_size, self.kernel_size), prob_vector.reshape(n, L, -1), prob_vector_previous.reshape(n, L, -1), 
                            origin_dot_product.reshape(n, L, -1, self.kernel_size, self.kernel_size))
        elif (float(self.which_mask) >= 2.0) and (float(self.which_mask) < 5.0):
            vc, sim_map_max = self.generate_mask(x) # [n * L, c], [n * L, 1, h, w]
            integrate_input = torch.cat([vc, sim_map_max.view(n * L, -1)], dim=1)
            # print(f"input *************************** {integrate_input.sum()}")
            integrated_output = self.integrate_mask_activation(integrate_input) # [n * L, c]
            # print(f"integreated mask activation ===================== {integrated_output.mean()}")
            return integrated_output.view(-1, L, c), None  # [n, L, c]
        elif (float(self.which_mask) >= 5.0) and (float(self.which_mask) < 6.0):
            # introduce maximun entropy 
            # print(f"===== test? {self.test}")
            vc, sim_map_max, neg_entorpy = self.generate_mask(x, test=self.test) # [n * L, c], [n * L, 1, h, w], scaler
            # print(f"sim_map_max {sim_map_max.shape}")
            # print(f"vc {vc.shape}")
            integrate_input = torch.cat([vc, sim_map_max.view(-1, self.kernel_size * self.kernel_size)], dim=1)
            # print(f"input *************************** {integrate_input.sum()}")
            integrated_output = self.integrate_mask_activation(integrate_input) # [n * L, c]
            # print(f"integreated mask activation ===================== {integrated_output.mean()}")
            return integrated_output.view(-1, L, c), neg_entorpy  # [n, L, c], scaler
        elif float(self.which_mask) >= 6.0:
            # introduce maximun entropy 
            vc, sim_map_max, neg_entorpy = self.generate_mask(x, test=self.test) # [n * L, c], None, scaler
            
            # print(f"integreated mask activation ===================== {integrated_output.mean()}")
            return vc.view(-1, L, c), neg_entorpy  # [n, L, c], scaler

 







#################################################
#          Conv Select main                     #
#################################################
class SparseNeuralConv(nn.Module):
    """main class to conv select and conv reconstruct"""
    def __init__(self, topk, ch, resolution, kernel_size, vc_dict_size, no_attention_select=False, sparse_vc_interaction=0, sparse_vc_prob_interaction=4, mode="1.0",
                        test=False):
        super(SparseNeuralConv, self).__init__()
        self.myid = "conv_sparse_vc_recover"
        self.mode = mode
        self.test = test
        
        if (float(self.mode) < 4.0) or (float(self.mode) >= 5.0 and float(self.mode) < 7.0):
            self.select = NeuralConvSelection(ch, resolution, kernel_size, vc_dict_size, no_attention=no_attention_select, which_mask=mode, \
                                                sparse_vc_prob_interaction=sparse_vc_prob_interaction, \
                                                test=self.test)
            self.recon = NeuralConvRecon(ch, resolution, kernel_size)
            self.sparse_vc_interaction = sparse_vc_interaction
            if self.sparse_vc_interaction:
                self.attention_modules = nn.ModuleList([BatchedVectorAttention(ch, max(ch // 5, 1)) for _ in range(self.sparse_vc_interaction)])
        
        elif float(self.mode) >= 4.0 and float(self.mode) < 5.0:
            """use multiple pathway to help with"""
            # for objects
            self.select1 = NeuralConvSelection(ch, resolution, kernel_size, vc_dict_size, no_attention=no_attention_select, which_mask=mode, \
                                                sparse_vc_prob_interaction=sparse_vc_prob_interaction,
                                                vc_type="parts")
            self.recon1 = NeuralConvRecon(ch, resolution, kernel_size)
            self.sparse_vc_interaction = sparse_vc_interaction
            if self.sparse_vc_interaction:
                self.attention_modules1 = nn.ModuleList([BatchedVectorAttention(ch, max(ch // 5, 1)) for _ in range(self.sparse_vc_interaction)])
            
            # for texture
            self.select2 = NeuralConvSelection(ch, resolution, kernel_size, vc_dict_size, no_attention=no_attention_select, which_mask=mode, \
                                                sparse_vc_prob_interaction=sparse_vc_prob_interaction,
                                                vc_type="texture")
            self.recon2 = NeuralConvRecon(ch, resolution, kernel_size)
            self.sparse_vc_interaction = sparse_vc_interaction
            if self.sparse_vc_interaction:
                self.attention_modules2 = nn.ModuleList([BatchedVectorAttention(ch, max(ch // 5, 1)) for _ in range(self.sparse_vc_interaction)])


    def forward(self, x, eval_=False, select_index=0, device="cuda"):
        """
        x: [n, c, h, w]
        output: [n, c, h, w]
        """
        ####### 1.0 #########
        if self.mode == '1.0':
            vcs, (mask_x, prob_vector, previous_prob_vector, origin_map) = self.select(x) # [n, L, c]
        
            if self.sparse_vc_interaction:
                pass

            if eval_:
                n, L, c = vcs.shape
                if type(select_index) == int:
                    select_index = [select_index]
                select_indexs = select_index # if len(select_index) >= 2 else range(select_index[0])
                select_indexs = torch.LongTensor([i for i in select_indexs]).view(-1, 1).unsqueeze(0).repeat(n, 1, c).to(device)
                # index = (torch.ones(c, dtype=torch.long) * select_index).unsqueeze(0).unsqueeze(0).expand((n,1,c)).to(device)
                vcs = vcs * torch.zeros_like(vcs).scatter_(1, select_indexs, 1.).to(device)
                print(f"Mask out vcs, testing vc {select_index}")

            x = self.recon(vcs) # [n, c, h, w]
            return x, (mask_x, prob_vector, previous_prob_vector, origin_map)
        
        ####### 2.0+ & 3.0+ #########
        elif float(self.mode) >= 2.0 and float(self.mode) < 4.0:

            vcs, _ = self.select(x) # [n, L, c]
            if self.sparse_vc_interaction:
                for attend in self.attention_modules:
                    vcs = attend(vcs) # [n, L, c]
            x = self.recon(vcs) # [n, c, h, w]
            return x, None
        elif float(self.mode) >= 4.0 and float(self.mode) < 5.0:
            # first parts
            vcs, _ = self.select1(x) # [n, L, c]
            if self.sparse_vc_interaction:
                for attend in self.attention_modules1:
                    vcs = attend(vcs) # [n, L, c]
            x1 = self.recon1(vcs) # [n, c, h, w]

            # second parts
            vcs, _ = self.select2(x) # [n, L, c]
            if self.sparse_vc_interaction:
                for attend in self.attention_modules2:
                    vcs = attend(vcs) # [n, L, c]
            x2 = self.recon2(vcs) # [n, c, h, w]

            x = x1 + x2 
            return x, None # [n, c, h, w]
        elif float(self.mode) >= 5.0 and float(self.mode) < 7.0:
            # in 5.0, introduce maximum entropy princinple
            # print("IN 5.0")
            vcs, neg_entorpy = self.select(x) # [n, L, c]
            # print("neg_entorpy", neg_entorpy.item())
            if self.sparse_vc_interaction:
                for attend in self.attention_modules:
                    vcs = attend(vcs) # [n, L, c]
            x = self.recon(vcs) # [n, c, h, w]
            return x, neg_entorpy





        



