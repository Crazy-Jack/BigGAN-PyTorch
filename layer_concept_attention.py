import torch 
import torch.nn as nn 
import torchvision 
import os



class ConceptPool(nn.Module):
    """concept pool that contains multiple concepts"""
    def __init__(self, num_k=20, pool_size_per_cluster=100, feature_dim=128):
        






class ConceptAttentionProto(nn.Module):
    """concept attention with prototype to reduce attention time"""
    def __init__(self, topk, visual_concept_pool_size, visual_concept_dim, mode, lambda_l1_reg_dot=1, test=False):
        super(ConceptAttentionProto, self).__init__()
        self.register_buffer('', torch.zeros(K, dtype=torch.long))
        
    def forward(self, x, device="cuda"):
        n, c, h, w = x.shape 

        # 





