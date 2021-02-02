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
#               Diff Mask module                #
#################################################

class VectorAttention(nn.Module):
    """vector attention"""
    def __init__(self, input_dim, hidden_dim):
        super(VectorAttention, self).__init__()
        self.theta = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(input_dim, hidden_dim)
        self.psi = nn.Linear(input_dim, hidden_dim)
        self.recover = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        x_t = self.theta(x) # [n, hidden_dim]
        x_ph = self.phi(x) # [n, hidden_dim]
        x_psi = self.psi(x) # [n, hidden_dim]

        attention_map = torch.matmul(x_ph, torch.transpose(x_t, 0, 1)) # n, n
        # TODO: integrate with position information
        attention_map = attention_map # [n, n]
        attention_map = F.softmax(attention_map, dim=1) # normalize, [n, n]
        x_add = torch.matmul(attention_map, x_psi) # [n, hidden_dim]
        x_add = self.recover(x_add) # [n, input_dim]
        return x + x_add

class BatchedVectorAttention(nn.Module):
    """vector attention"""
    def __init__(self, input_dim, hidden_dim):
        super(BatchedVectorAttention, self).__init__()
        self.theta = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(input_dim, hidden_dim)
        self.psi = nn.Linear(input_dim, hidden_dim)
        self.recover1 = nn.Linear(hidden_dim, max(input_dim // 2, 1))
        self.lrelu = nn.LeakyReLU(0.2)
        self.recover2 = nn.Linear(max(input_dim // 2, 1), input_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """
        x: [n, L, c]
        """
        n, L, c = x.shape
        x_reshape = x.view(-1, c) # [n * L, c]
        x_t = self.theta(x).view(n, L, -1) # [n, L, hidden_dim]
        x_ph = self.phi(x).view(n, L, -1) # [n, L, hidden_dim]
        x_psi = self.psi(x).view(n, L, -1) # [n, L, hidden_dim]

        attention_map = torch.matmul(x_ph, torch.transpose(x_t, 1, 2)) # n, L, L
        # TODO: integrate with position information
        attention_map = attention_map # n, L, L
        attention_map = F.softmax(attention_map, dim=2) # normalize, n, L, L
        x_add = torch.matmul(attention_map, x_psi) # [n, L, L] x [n, L, hidden_dim] => [n, L, hidden_dim]
        
        # recover
        x_add = self.recover1(x_add.view(n * L, -1)) # [n * L, hidden_dim] -> [n * L, input_dim // 2]
        x_add = self.lrelu(x_add)
        x_add = self.recover2(x_add) # [n * L, input_dim // 2] -> [n * L, input_dim]
        x_add = self.tanh(x_add) # [n * L, input_dim]
        x_add = x_add.view(n, L, c)
        return x + x_add # [n, L, input_dim]

class GenerateMask(nn.Module):
    """
    sparsify by 1x1 conv and apply gumbel softmax
    """
    def __init__(self, ch, resolution, or_cadidate=1000, no_attention=False):
        super(GenerateMask, self).__init__()
        self.conv = nn.Conv2d(ch, or_cadidate, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.no_attention = no_attention
        if not self.no_attention:
            self.prob_attention = VectorAttention(input_dim=or_cadidate, hidden_dim=or_cadidate//3)

    def forward(self, x):
        """
        input x: [n, L, c, kernel, kernel]
        output mask: [n * L, 1, h, w]
        """
        n, L, c, h, w = x.shape
        # print(f"x {x.shape}")
        y = self.conv(x.reshape(-1, c, h, w)) # [n * L, or_cadidate, h, w]
        prob_vector = F.softmax(y.sum((2,3)), dim=1) # n * L, or_cadidate
        prob_vector_previous = prob_vector#y.sum((2,3))# prob_vector
        # TODO: generating position nn and incorporate position encoding with the attention model
        # position_nn = 

        # interact between prob_vector to reduce ambiguity
        if not self.no_attention:
            prob_vector = self.prob_attention(prob_vector) # n * L, or_cadidate
        prob_vector = F.softmax(prob_vector) # n * L, or_cadidate

        # generate weighted mask
        weighted_mask = (prob_vector[:, :, None, None] * y).sum((1,), keepdim=True) # [n * L, or_cadidate, h, w] -> [n * L, 1, h, w]
        weighted_mask = F.softmax(weighted_mask.view(n * L, 1, h*w), dim=2).view(n * L, 1, h, w) # [n * L, 1, h, w], normalize in image
        
        return weighted_mask, prob_vector, prob_vector_previous, y
        

class GenerateMask_2_0(nn.Module):
    """
    sparsify by 1x1 conv and apply gumbel softmax; 2.0 version
    new: 
        - when generating the mask, do normalization of the dot product to get the real similarity
        - when 

    """
    def __init__(self, ch, resolution, or_cadidate=1000):
        super(GenerateMask_2_0, self).__init__()
        self.conv = nn.Conv2d(ch, or_cadidate, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.or_cadidate = or_cadidate
      

    def forward(self, x):
        """
        input x: [n, L, c, kernel, kernel]
        output mask: [n * L, 1, kernel, kernel]
        """
        n, L, c, h, w = x.shape
        x_reshape = x.reshape(-1, c, h, w) # [n * L, c, h, w]
        x_norm = (x_reshape ** 2).sum((1)).sqrt() # [n * L, h, w]
        weights = self.conv.weight # [or_cadidate, c, 1, 1]
        weights_norm = (weights**2).sum(1).sqrt().view(self.or_cadidate,) # [or_cadidate,]

        # print(f"x {x.shape}")
        y = self.conv(x_reshape) # [n * L, or_cadidate, h, w]
        # normalize the similarity map: 
        y = y / (x_norm[:, None, :, :] * weights_norm[None, :, None, None]) # normalized by it length: [n * L, or_cadidate, h, w] range: (-1, 1)

        # get max channel index
        max_similarity_value, index = torch.topk(y.view(n*L, -1), 1, dim=1) # max_similarity_value: [n * L, 1]
        max_ch_index = index[:, 0] // (h * w) # [n*L, ], find the max channel index by playing around the index since we do y.view(, -1), all the c, h, w becomes a linear vector , out of total vc numbers

        # get max vc similarity map
        max_ch_index = max_ch_index.unsqueeze(1) # [n*L, 1]
        gather_index_max_vc_map = max_ch_index.unsqueeze(2).unsqueeze(2).expand(max_ch_index.size(0), max_ch_index.size(1), h, w) 
        sim_map_max = torch.gather(y, 1, gather_index_max_vc_map) # n * L, 1, h, w
        
        # get the max activation column
        max_activation_column_index = index[:, 0] % (h * w)  # [n*L, ], 

        index_0 = torch.LongTensor([[i for _ in range(c)] for i in range(n*L)]) # [[0,0,0,0,0], [1,1,1,1,1]] if c = 5
        index_1 = torch.LongTensor([[i for i in range(c)] for _ in range(n*L)]) # [[0,1,2,3,4],[0,1,2,3,4]] if n*L = 2
        index_2 = max_activation_column_index.view(-1, 1).expand(n*L, c) # [[3,3,3,3,3],[2,2,2,2,2]]] if n*L = 2
        max_activation = x_reshape.view(n * L, c, -1)[index_0, index_1, index_2] # n * L, c

        # get corresponding vc 
        vc = torch.gather(weights.squeeze(2).squeeze(2), 0, max_ch_index.expand(max_ch_index.size(0), c)) # [n * L, c]

        # intepolate vc - activation
        integrate_activation_vc = max_activation * max_similarity_value + vc * (1 - max_similarity_value) # integrate_activation_vc = activation * sim + vc * (1 - sim): [n * L, c]

        return integrate_activation_vc, sim_map_max # [n * L, c], [n * L, 1, h, w]

        

#################################################
#         idea: convolutional selection         #
#################################################
"""perform hypercolumn sparsity since within each scope"""

class NeuralConvSelection(nn.Module):
    """conv select"""
    def __init__(self, ch, resolution, kernel_size, vc_dict_size=100, no_attention=False, hard_selection=False, which_mask="1.0"):
        super(NeuralConvSelection, self).__init__()
        self.kernel_size = kernel_size
        self.which_mask = which_mask
        if which_mask == '1.0':
            self.generate_mask = GenerateMask(ch, resolution=kernel_size, or_cadidate=vc_dict_size, no_attention=no_attention) # shared generate mask module, generate a weighted mask for 
        elif float(which_mask) >= 2.0:
            self.generate_mask = GenerateMask_2_0(ch, resolution=kernel_size, or_cadidate=vc_dict_size) # shared generate mask module, generate a weighted mask for 
            self.integrate_mask_activation = nn.Sequential(
                nn.Linear(ch + kernel_size * kernel_size, int(ch * 3)),
                nn.LeakyReLU(0.2),
                nn.Linear(int(ch * 3), ch),
                nn.Tanh()
            )
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
            x = x.view(n, -1, c) # [n, L, c]
            return x, (x_mask.reshape(n, L, self.kernel_size, self.kernel_size), prob_vector.reshape(n, L, -1), prob_vector_previous.reshape(n, L, -1), 
                            origin_dot_product.reshape(n, L, -1, self.kernel_size, self.kernel_size))
        elif float(self.which_mask) >= 2.0 and float(self.which_mask) < 3.0:
            vc, sim_map_max = self.generate_mask(x) # [n * L, c], [n * L, 1, h, w]
            integrate_input = torch.cat([vc, sim_map_max.view(n * L, -1)], dim=1)
            integrated_output = self.integrate_mask_activation(integrate_input) # [n * L, c]

            return integrated_output.view(n, -1, c), None  # [n, L, c]
 




#################################################
#          Reconstruct convolutionally          #
#################################################
class NeuralConvRecon(nn.Module):
    """conv reconstruct by mlp"""
    def __init__(self, ch, resolution, kernel_size):
        super(NeuralConvRecon, self).__init__()
        self.kernel_size = kernel_size
        self.linear1_out = (ch // kernel_size) * (kernel_size * kernel_size) 
        self.conv_ch_in = ch // kernel_size
        self.interm_ch = self.linear1_out // (kernel_size * kernel_size)
        # define architecture
        self.linear1 = nn.Linear(ch, self.linear1_out // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.linear1_out // 2, self.linear1_out)
        self.conv = nn.Conv2d(self.conv_ch_in, ch, kernel_size=1, padding=0, bias=False)
        self.relu2 = nn.ReLU()

        # fold
        self.fold = nn.Fold(output_size=(resolution, resolution), kernel_size=(kernel_size, kernel_size), stride=3)

    def forward(self, x):
        """
        x: [n, L, c]
        [n, L, c] -> [n, L, c * kernel * kernel] -> [n, c * kernel * kernel, L] -> fold => [n, c, h, w]
        output: [n, c, h, w]
        """
        n, L, c = x.shape
        x = self.linear1(x.view(-1, c)) # [n * L, c] -> [n * L, self.linear1_out // 2]
        x = self.relu1(x)
        x = self.linear2(x) # [n * L, self.linear1_out]
        x = x.view(n * L, -1, self.kernel_size, self.kernel_size) # [n * L, ch // kernel_size, kernel_size, kernel_size]
        x = self.conv(x) # [n * L, ch, kernel_size, kernel_size]
        x = x.view(n, L, c * self.kernel_size * self.kernel_size)
        x = torch.transpose(x, 1, 2) # [n, c * kernel * kernel, L]
        x = self.fold(x) # n, c, h, w

        return x


class Deconvolution(nn.Module):
    """reconstruct by deconvolution to introduce conv prior"""
    def __init__(self, ):
        pass

    def forward(self, x):
        """
        x: [n, L, c]
        [n, L, c] -> [n, L, c * kernel * kernel] -> [n, c * kernel * kernel, L] -> fold => [n, c, h, w]
        output: [n, c, h, w]
        """
        pass



#################################################
#          Conv Select main                     #
#################################################
class SparseNeuralConv(nn.Module):
    """main class to conv select and conv reconstruct"""
    def __init__(self, topk, ch, resolution, kernel_size, vc_dict_size, no_attention_select=False, sparse_vc_interaction=0, mode="1.0"):
        super(SparseNeuralConv, self).__init__()
        self.myid = "conv_sparse_vc_recover"
        self.mode = mode
        self.select = NeuralConvSelection(ch, resolution, kernel_size, vc_dict_size, no_attention=no_attention_select, which_mask=mode)
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
        
        ####### 2.0 #########
        elif float(self.mode) >= 2.0:
            vcs, _ = self.select(x) # [n, L, c]
            if self.sparse_vc_interaction:
                for attend in self.attention_modules:
                    vcs = attend(vcs) # [n, L, c]
            x = self.recon(vcs) # [n, c, h, w]
            return x, None

        



