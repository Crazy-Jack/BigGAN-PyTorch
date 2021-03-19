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






#################################################
#          Conv Select main                     #
#################################################
class SparseNeuralConvMulti(nn.Module):
    """main class to conv select and conv reconstruct"""
    def __init__(self, topk, ch, resolution, kernel_size, vc_dict_size, no_attention_select=False, sparse_vc_interaction=0, sparse_vc_prob_interaction=4, mode="1.0"):
        super(SparseNeuralConvMulti, self).__init__()
        self.myid = "conv_sparse_vc_recover"
        self.mode = mode
        self.select = NeuralConvSelection(ch, resolution, kernel_size, vc_dict_size, no_attention=no_attention_select, which_mask=mode, sparse_vc_prob_interaction=sparse_vc_prob_interaction)
        self.recon = NeuralConvRecon(ch, resolution, kernel_size)
        self.sparse_vc_interaction = sparse_vc_interaction
        if self.sparse_vc_interaction:
            self.attention_modules = nn.ModuleList([BatchedVectorAttention(ch, max(ch // 5, 1)) for _ in range(self.sparse_vc_interaction)])
            
    
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
        elif float(self.mode) >= 2.0:

            vcs, _ = self.select(x) # [n, L, c]
            if self.sparse_vc_interaction:
                for attend in self.attention_modules:
                    vcs = attend(vcs) # [n, L, c]
            x = self.recon(vcs) # [n, c, h, w]
            return x, None
