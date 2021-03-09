import torch 
import torch.nn as nn 
from kornia.filters.kernels import (
    get_spatial_gradient_kernel2d, get_spatial_gradient_kernel3d, normalize_kernel2d
)

class SparseGradient(nn.Module):
    """instead of taking the topk on the neurons activation, what if we take the sparsity based on the gradient of the neurons"""
    def __init__(self, topk):
        super(SparseGradient, self).__init__()
        self.topk = topk 
    
    def forward(self, x):
        