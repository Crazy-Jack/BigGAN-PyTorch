

import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.autograd import Variable
from torch.distributions import Categorical


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
        self.doc_vc_statistics = {i: 0 for i in range(or_cadidate)}
        self.total_num_instance = 0

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

        # test specific vc
        ####### START testing #####
        ## fast prototype code
        # return [n * L, c] where only every n's first vector is target vector and the sim_map is 
        # get statistics

        if True:
            # self.doc_used_vc(max_ch_index)
            # visualize vc
            return self.visualize_vc(n, L, c, h, w)


        ####### END testing ######
        
        return integrate_activation_vc, sim_map_max # [n * L, c], [n * L, 1, h, w]
    
    def doc_used_vc(self, max_ch_index):
        for index in max_ch_index.view(max_ch_index.shape[0]):
            self.doc_vc_statistics[index.item()] += 1
        
        
        total_num = sum([self.doc_vc_statistics[i] for i in self.doc_vc_statistics])
        print(f"total NOW  !!!!!******* {total_num} *********** !!!")
        import json 
        with open("/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/celeba/analysis/vc_statistics.json", 'w') as f:
            json.dump(self.doc_vc_statistics, f, indent=4, sort_keys=True)
            print("dump")


    def visualize_vc(self, n, L, c, h, w):
        indexs = [4423, 4793, 295, 3307]
        weights = self.conv.weight.squeeze(2).squeeze(2) # [or_cadidate, c]
        mask = torch.zeros((n, L, c)).to("cuda")
        mapping = torch.zeros((n, L, h, w)).to("cuda")
        for idx, index in enumerate(indexs):
            target_vc =  weights[index]
            mask[idx, L-1, :] = target_vc
            mapping[idx, L-1, h//2, w//2] = 1. # [n * L, 1, h, w]
        mask = mask.view(n * L, c) # n * L, c
         
        print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  Sum {mask.sum()}, {mapping.sum()}")
        return mask, mapping




class GenerateMask_3_0(nn.Module):
    """
    sparsify by 1x1 conv and apply gumbel softmax; 3.0 version
    new: 
        - when generating the mask, do normalization of the dot product to get the real similarity
        - do self attention helped vc selection before output
    """
    def __init__(self, ch, resolution, or_cadidate=1000, sparse_vc_prob_interaction=20, vc_type="parts", tmp=0.05, reg_entropy=False, warmup=5436000, no_map=False, \
                 pull_vc_activation=False, replace_activation=False):
        super(GenerateMask_3_0, self).__init__()
        self.conv = nn.Conv2d(ch, or_cadidate, kernel_size=1, padding=0, bias=False)
        self.sparse_vc_prob_interaction = sparse_vc_prob_interaction
        self.vc_prob_attention = nn.ModuleList([BatchedVectorAttention(or_cadidate, max(or_cadidate // 5, 1)) for _ in range(self.sparse_vc_prob_interaction)])
        self.relu = nn.ReLU()
        self.or_cadidate = or_cadidate
        # define types of vc
        assert vc_type in ['parts', 'texture']
        self.vc_type = vc_type
        self.tmp = tmp
        self.no_map = no_map
        if reg_entropy:
            self.warmup = warmup
            self.reg_entropy = reg_entropy
            print("reg_entropy", reg_entropy)

            self.register_buffer("vc_stats", torch.FloatTensor([0.] * self.or_cadidate))
        self.pull_vc_activation = pull_vc_activation # False, or (0.1) indicating the beta value, (0) means beta is 0 but the error for stored vc is not zero
        print(f"pull vc_activation {self.pull_vc_activation}")
        self.replace_activation = replace_activation

    def forward(self, x, test=False, device="cuda"):
        """
        input x: [n, L, c, kernel, kernel]
        output mask: [n * L, 1, kernel, kernel]
        """
        # additional term to regularize
        additional_loss = []

        n, L, c, h, w = x.shape
        x_reshape = x.reshape(-1, c, h, w) # [n * L, c, h, w]
        x_norm = (x_reshape ** 2).sum((1)).sqrt() # [n * L, h, w]
        weights = self.conv.weight # [or_cadidate, c, 1, 1]
        weights_norm = (weights**2).sum(1).sqrt().view(self.or_cadidate,) # [or_cadidate,]

        # print(f"x {x.shape}")
        y = self.conv(x_reshape) # [n * L, or_cadidate, h, w]
        # normalize the similarity map: 
        y = y / (x_norm[:, None, :, :] * weights_norm[None, :, None, None]) # normalized by it length: [n * L, or_cadidate, h, w] range: (-1, 1)
        
        if self.vc_stats.sum() < self.warmup:
            print(f"Warmup sum of vc_stats {self.vc_stats.sum()} < {self.warmup} ")
            y += Variable(torch.rand(y.shape).to(device))

        # batched max response of each vc response
        if self.vc_type == 'parts':
            # print(f"n {n}, L {L}, c {c}, h {h}, w {w}")
            # print(f"y.shape {y.shape}")
            max_y = torch.max(y.reshape(n * L, -1, h * w), dim=2)[0] # [n * L, or_cadidate]
        elif self.vc_type == 'texture':
            max_y = F.softmax(1 / (e-12 + y.std((2, 3))) / self.tmp, dim=1) # [n * L, or_cadidate]
        else:
            raise NotImplementedError(f"self.vc_type {self.vc_type} is not implemented in GenerateMask_4_0 forward path")

        max_y = max_y.reshape(n, L, -1) # [n, L, or_cadidate]
        # attention modules for selecting vc
        for attention in self.vc_prob_attention:
            max_y = attention(max_y) # tanh at the end of each attention layer <-1, 1>; max_y : [n, L, or_cadidate]
        
        # get max channel index
        _, index = torch.topk(max_y.view(n*L, -1), 1, dim=1) # max_similarity_value: [n * L, 1]

        # get max vc similarity map
        max_ch_index = index.view(-1, 1) # [n * L, 1]
        
        gather_index_max_vc_map = max_ch_index.unsqueeze(2).unsqueeze(2).expand(max_ch_index.size(0), max_ch_index.size(1), h, w) 
        sim_map_max = torch.gather(y, 1, gather_index_max_vc_map) # n * L, 1, h, w

        if self.reg_entropy:
            # store which vc being used
            for vc in max_ch_index.view(-1):
                self.vc_stats[vc] += 1 
        
        # get the max activation column
        max_similarity_value, max_activation_column_index = torch.topk(sim_map_max.view(n * L, -1), 1, dim=1)
        max_activation_column_index = max_activation_column_index.squeeze(1)  # [n*L, ], 

        index_0 = torch.LongTensor([[i for _ in range(c)] for i in range(n*L)]) # [[0,0,0,0,0], [1,1,1,1,1]] if c = 5
        index_1 = torch.LongTensor([[i for i in range(c)] for _ in range(n*L)]) # [[0,1,2,3,4],[0,1,2,3,4]] if n*L = 2
        index_2 = max_activation_column_index.view(-1, 1).expand(n*L, c) # [[3,3,3,3,3],[2,2,2,2,2]]] if n*L = 2
        max_activation = x_reshape.view(n * L, c, -1)[index_0, index_1, index_2] # n * L, c

        # get corresponding vc 
        vc = torch.gather(weights.squeeze(2).squeeze(2), 0, max_ch_index.expand(max_ch_index.size(0), c)) # [n * L, c]

        # intepolate vc - activation
        if self.replace_activation:
            integrate_activation_vc = vc
        else:
            integrate_activation_vc = max_activation * max_similarity_value + vc * (1 - max_similarity_value) # integrate_activation_vc = activation * sim + vc * (1 - sim): [n * L, c]
        if self.pull_vc_activation:
            # pull vc towards activation
            beta = self.pull_vc_activation[0]
            # error for vc
            error_vc = ((max_activation.detach() - vc) ** 2).mean()
            error_activation = beta * ((max_activation - vc.detach()) ** 2).mean()
            error_pull_vc_activation = error_vc + error_activation
            additional_loss.append(error_pull_vc_activation)

        # print("-----------------CHECKPOINT ---------------")
        if test:
            print("Testing vc...")
            print("VC useage statistics")
            uniques_t = {}
            for i in max_ch_index.view(-1):
                if i.item() not in uniques_t:
                    uniques_t[i.item()] = 0
                else:
                    uniques_t[i.item()] += 1
            print(f"One batch {uniques_t}")
            print(f"sim score {max_similarity_value.mean()}")
            integrate_activation_vc, sim_map_max = test_vc(integrate_activation_vc, max_ch_index, sim_map_max, L, c, h, w)

        
        # maximize negative entropy of the y
        if self.reg_entropy:
            p = self.vc_stats / self.vc_stats.sum()
            print(f"VC concepts prob: mean {p.mean().item():5f} std {p.std().item():5f}; <0.5 {len(torch.where(p<0.5)[0])};"
                        f"<0.05 {len(torch.where(p<0.05)[0])} ; <0.01 {len(torch.where(p<0.01)[0])} ;" 
                        f"<0.005 {len(torch.where(p<0.005)[0])};"
                        f"<0.001 {len(torch.where(p<0.001)[0])};"
                        f"<0.0005 {len(torch.where(p<0.0005)[0])};")
            neg_entropy_vc_stats = - Categorical(probs = p).entropy() # negative entropy
            additional_loss.append(neg_entropy_vc_stats)
            # print(f"Entropy {neg_entropy_vc_stats}")

        # integrate additional loss
        if len(additional_loss) > 0:
            additional = additional_loss[0]
            if len(additional_loss) > 1:
                for addi_loss in additional_loss[1:]:
                    additional += addi_loss
        else:
            additional = None 

        if self.no_map:
            sim_map_max = None

        # regularize entropy term
        if additional:
            return integrate_activation_vc, sim_map_max, additional # [n * L, c], [n * L, 1, h, w], scaler

        return integrate_activation_vc, sim_map_max # [n * L, c], [n * L, 1, h, w]



def test_vc(integrate_activation_vc, max_ch_index, sim_map_max, L, c, h, w):
    """function to isolate individual vc
    param: 
        integrate_activation_vc: [n * L, c]
        sim_map_max: [n * L, 1, h, w]
        max_ch_index: [n * L, 1]
    output: 
        [(nk+n) * L, c] where nk is sum of number of vc used in each image
        [(nk+n) * L, 1, h, w] corresponding to the mask
    """
    # get activation for each seperated vc
    integrate_activation_vc = integrate_activation_vc.view(-1, L, c) # n, L, c
    sim_map_max = sim_map_max.view(-1, L, 1, h, w) # n, L, 1, h, w
    max_ch_index = max_ch_index.view(-1, L) # n, L

    all_activation_vcs = []

    for i in range(len(integrate_activation_vc)):
        integrate_activation_vc_i = integrate_activation_vc[i] # L, c
        all_activation_vcs.append(integrate_activation_vc_i) 
        max_ch_index_i = max_ch_index[i] # L,
        unique_vc_i = torch.unique(max_ch_index_i) # k,
        print(f"unique_vc_i {unique_vc_i.shape}")
        for vc_i in unique_vc_i:
            index_vc_i = torch.where(max_ch_index_i != vc_i)[0] # num of miss hit for that vc out of L
            activate_vc_i = integrate_activation_vc_i.clone() # L, c
            activate_vc_i[index_vc_i, :] = 0.
            all_activation_vcs.append(activate_vc_i)
    
    all_activation_vcs = torch.cat(all_activation_vcs, dim=0) # [(nk+n) * L, c]
    
    # decompose each mask 
    all_masks = []
    for i in range(len(sim_map_max)):
        sim_map_max_i = sim_map_max[i] # L, 1, h, w
        all_masks.append(sim_map_max_i) 
        max_ch_index_i = max_ch_index[i] # L,
        unique_vc_i = torch.unique(max_ch_index_i) # k,
        for vc_i in unique_vc_i:
            index_vc_i = torch.where(max_ch_index_i != vc_i)[0] # num of miss hit for that vc out of L
            vc_sim_map_max_i = sim_map_max_i.clone() # L, 1, h, w
            vc_sim_map_max_i[index_vc_i] = 0.
            all_masks.append(vc_sim_map_max_i)
    all_masks = torch.cat(all_masks, dim=0) # [(nk+n) * L, 1, h, w]

    assert all_masks.shape[0] == all_activation_vcs.shape[0], f"all_masks.shape[0] {all_masks.shape[0]} != all_activation_vcs.shape[0] {all_activation_vcs.shape[0]}"

    return all_activation_vcs, all_masks




            



    
    




    



