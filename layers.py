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

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d

from layer_conv_select import SparseNeuralConv
from layer_vc_linear_comb import LinearCombineVC
from layer_conv_select_multiple_path import SparseNeuralConvMulti
from layer_concept_attention_proto import ConceptAttentionProto
from layer_concept_attention_moca import MomemtumConceptAttentionProto
from torch.distributions import Categorical

# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, input):
        return input


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(
                W_mat, self.u, update=self.training, eps=self.eps)
        # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
        self.myid = "snconv2d"

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
        self.myid = "snlinear"

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq,
                              sparse, _weight)
        SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)
        self.myid = "snembedding"

    def forward(self, x):
        return F.embedding(x, self.W_())


# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        self.myid = "atten"
        
        # Channel multiplier
        self.ch = ch
        print(f"INSIDE ATTENTION   self.ch // 2 {self.ch // 2}")
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 2, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift
    # return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = (m2 - m ** 2)
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super(myBN, self).__init__()
        self.myid = "mybn"
        # momentum for updating running stats
        self.momentum = momentum
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Register buffers
        self.register_buffer('stored_mean', torch.zeros(num_channels))
        self.register_buffer('stored_var',  torch.ones(num_channels))
        self.register_buffer('accumulation_counter', torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(
                x, gain, bias, return_mean_var=True, eps=self.eps)
            # If accumulating standing stats, increment them
            if self.accumulate_standing:
                self.stored_mean[:] = self.stored_mean + mean.data
                self.stored_var[:] = self.stored_var + var.data
                self.accumulation_counter += 1.0
            # If not accumulating standing stats, take running averages
            else:
                self.stored_mean[:] = self.stored_mean * \
                    (1 - self.momentum) + mean * self.momentum
                self.stored_var[:] = self.stored_var * \
                    (1 - self.momentum) + var * self.momentum
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
    # If number of channels specified in norm_style:
    if 'ch' in norm_style:
        ch = int(norm_style.split('_')[-1])
        groups = max(int(x.shape[1]) // ch, 1)
    # If number of groups specified in norm style
    elif 'grp' in norm_style:
        groups = int(norm_style.split('_')[-1])
    # If neither, default to groups = 16
    else:
        groups = 16
    return F.group_norm(x, groups)


# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False, norm_style='bn',):
        super(ccbn, self).__init__()
        self.myid = "ccbn"
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # Norm style?
        self.norm_style = norm_style

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps,
                               momentum=self.momentum, affine=False)
        elif self.mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
        elif self.norm_style in ['bn', 'in']:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',  torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        # If using my batchnorm
        if self.mybn or self.cross_replica:
            return self.bn(x, gain=gain, bias=bias)
        # else:
        else:
            if self.norm_style == 'bn':
                out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                                   self.training, 0.1, self.eps)
            elif self.norm_style == 'in':
                out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                      self.training, 0.1, self.eps)
            elif self.norm_style == 'gn':
                out = groupnorm(x, self.normstyle)
            elif self.norm_style == 'nonorm':
                out = x
            return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        s += ' cross_replica={cross_replica}'
        return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                 cross_replica=False, mybn=False):
        super(bn, self).__init__()
        self.myid = "bn"
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = P(torch.ones(output_size), requires_grad=True)
        self.bias = P(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # Use cross-replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn

        if self.cross_replica:
            self.bn = SyncBN2d(output_size, eps=self.eps,
                               momentum=self.momentum, affine=False)
        elif mybn:
            self.bn = myBN(output_size, self.eps, self.momentum)
           # Register buffers if neither of the above
        else:
            self.register_buffer('stored_mean', torch.zeros(output_size))
            self.register_buffer('stored_var',  torch.ones(output_size))

    def forward(self, x, y=None):
        if self.cross_replica or self.mybn:
            gain = self.gain.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return self.bn(x, gain=gain, bias=bias)
        else:
            return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                                self.bias, self.training, self.momentum, self.eps)


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()
        self.myid = "gblock"
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y, nobn=False):
        if nobn:
            h = self.activation(x)
        else:
            h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        if nobn:
            h = self.activation(x)
        else:
            h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
        self.myid = "dblock"
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (
            in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)

# dogball


class Sparsify_hw(nn.Module):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk  # percent of the top reponse to keep
        self.myid = "sparse_hw"

    def forward(self, x, tau, device='cuda'):
        n, c, h, w = x.shape
        x_reshape = x.view(n, c, h * w)
        keep_top_num = max(int(self.topk * h * w), 1)
        _, index = torch.topk(x_reshape.abs(), keep_top_num, dim=2)
        mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(device)
        # print("mask percent: ", mask.mean().item())
        
        sparse_x = mask * x_reshape
        sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
        print("sparsity -- ({}): {}".format((n, c, h, w), sparsity_x)) ## around 9% decrease to 4% fired eventually this way
        if tau == 1.0:
            return sparse_x.view(n, c, h, w)
        
        # print("--- tau", tau)
        tau_x = x * torch.FloatTensor([1. - tau]).to(device)
        # print("sum of x used", tau_x.sum())
        return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x


class Sparsify_ch(nn.Module):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk  # percent of the top reponse to keep
        self.myid = "sparse_ch"

    def forward(self, x, tau, device='cuda'):
        n, c, h, w = x.shape
        keep_top_num = max(int(self.topk * c), 1)
        _, index = torch.topk(x.abs(), keep_top_num, dim=1)
        mask = torch.zeros_like(x).scatter_(1, index, 1).to(device)
        # print("mask percent: ", mask.mean().item())
        sparse_x = mask * x
        if tau == 1.0:
            return sparse_x.view(n, c, h, w)

        tau_x = x * torch.FloatTensor([1. - tau]).to(device)
        # print("sum of x used", tau_x.sum())
        return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x



def sparsify_layer(h, tau, sparsity=0.1, device='cuda', mask_base=0.0):
    """Sparsify entire layer activation by sparsity level specified (within each instance)
    param: 
      - h: (N, C, H, W)
      - sparsity: float
    return: layer_concept_attention_proto
      sparse_activation: (N, C, H, W)
    """
    N, C, H, W = h.shape

    mask_base

    # get cutoff
    if sparsity != 1.0:
        cutoff = torch.sort(h.reshape(N, -1).abs(), descending=True)[0][:, int(sparsity * C * H * W)] # (N, )
        # print("keep : {}, cutoff max {}".format(int(sparsity * C * H * W), cutoff.max()))
        cutoff = cutoff.to(device)
        # mask only for values that large than corresponding threshold
        if mask_base == 0.0:
            mask = torch.where(h.abs() >= cutoff[:, None, None, None], torch.tensor([1.]).to(device), torch.tensor([0.]).to(device)).to(device) # (N, C, H, W)
        else:
            mask = torch.rand(N, C, H, W) * mask_base # (N, C, H, W)
            mask = mask.to(device)
            mask[torch.where(h.abs() >= cutoff[:, None, None, None])] = 1.0
        # print("mask 1 number {} (h={})".format(mask.sum(), h.shape))
        if tau == 1.0:
            return mask * h
        return mask * h * torch.FloatTensor([tau]).to(device) + h * torch.FloatTensor([1.0 - tau]).to(device)
    else:
        return h


class Sparsify_all(nn.Module):
    """sparse modules that sparsify without distinguish channel and spatial"""
    def __init__(self, sparsity, mask_base):
        '''sparsity: float'''
        super(Sparsify_all, self).__init__()
        self.sparsity = sparsity
        self.mask_base = mask_base
        self.myid = "sparse_all"
    def forward(self, h, tau):
        return sparsify_layer(h, tau=tau, sparsity=self.sparsity, mask_base=self.mask_base)



class Sparsify_hypercol(nn.Module):
    """sparse module for hypercolumn sparsity"""
    def __init__(self, topk, mode="hyper_col_mean", kernel_size=(5,5), which_conv=None, ch=None, resolution=None, hidden_ch=40):
        super(Sparsify_hypercol, self).__init__()
        self.topk = topk  # percent of the top reponse to keep
        self.mode = mode # dictate how topk column is selected: 1) based on mean response 2) based on 
        self.myid = "sparse_hyper"
        self.kernel = kernel_size
        self.hidden_ch = hidden_ch
        
        # use nn if mode is "hyper_col_nn":
        if self.mode in ["hyper_col_nn", "hyper_col_nn_remap"]:
            if which_conv:
                self.which_conv = which_conv
            if ch:
                self.ch = ch # note the sparsity operation doesn't change the output channel size
            if resolution:
                self.resolution = resolution
            if self.mode == "hyper_col_nn":
                self.map_nn = self.which_conv(self.ch, 1, kernel_size=1, padding=0, bias=False)
            if self.mode == "hyper_col_nn_remap":
                self.reduce_map = self.which_conv(self.ch, self.hidden_ch, kernel_size=1, padding=0, bias=False)
                self.map_nn = self.which_conv(self.hidden_ch, 1, kernel_size=1, padding=0, bias=False)
                self.recover_map = self.which_conv(self.hidden_ch, self.ch, kernel_size=1, padding=0, bias=False)
                # recover on the hidden space
                self.remap_in_dim = int(self.topk * (self.resolution ** 2)) * self.hidden_ch + (self.resolution ** 2) # the topk and the position encode for selected hypercolumn
                self.remap_linear1 = nn.Linear(self.remap_in_dim, self.remap_in_dim)
                self.relu = nn.ReLU()
                self.remap_linear2 = nn.Linear(self.remap_in_dim, (self.ch * self.resolution ** 2))


    def get_selection_stats(self, x_reshape):
        """get selection statistics
        param:
            - x_reshape: (n, c, h * w)
        return: [n, h * w], topk of column is selected based on this metrics
        """
        if self.mode in ["hyper_col_mean", "hyper_col_center_mean"]:
            return x_reshape.mean(1, keepdim=True)
        elif self.mode == "hyper_col_absmax":
            return x_reshape.abs().max(1, keepdim=True).values
    
    def transform(self, x):
        """input x: [n, c, h, w]
        output: [n, 1, h, w]
        """
        if self.mode == 'hyper_col_center_mean':
            return x.mean(1, keepdim=True)
        elif self.mode in ['hyper_col_nn', 'hyper_col_nn_remap']:
            return self.map_nn(x)
            

    def forward(self, x, tau, device='cuda'):
        n, c, h, w = x.shape
        x_reshape = x.view(n, c, h * w)
        # get topk hypercolumn index
        if self.mode in ["hyper_col_mean", "hyper_col_absmax"]:
            # build topk index based on mean response of the column
            x_reshape_colstats = self.get_selection_stats(x_reshape) # [n, 1, h * w]
            keep_top_num = max(int(self.topk * h * w), 1)
            _, index = torch.topk(x_reshape_colstats, keep_top_num, dim=2)
            mask = torch.zeros_like(x_reshape_colstats).scatter_(1, index, 1).to(device)
        elif self.mode in ["hyper_col_center_mean"]:
            with torch.no_grad():
                transformed_x = self.transform(x) # calculate the hypercolumn statistics based on non-linear/linear transform [n, 1, h, w]
                unfolded_x = F.unfold(transformed_x, self.kernel) # n, L, k1*k2*1
                keep_top_num = max(int(self.topk * unfolded_x.shape[2]), 1)
                _, index = torch.topk(unfolded_x, keep_top_num, dim=2)
                mask = torch.zeros_like(unfolded_x).scatter_(2, index, 1) 
                mask = F.fold(mask, (h, w), self.kernel) # n, 1, h, w
                mask = torch.clamp(mask, min=0.0, max=1.0) # n, L, k1*k2*1
                mask = mask.view(n, 1, h * w).detach().to(device)
        
        if self.mode not in ["hyper_col_nn", "hyper_col_nn_local", "hyper_col_nn_remap"]:
            sparse_x = mask * x_reshape # use None for broadcast the column dim
            with torch.no_grad():
                sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
                # print("sparsity -- ({}): {}".format((n, c, h, w), sparsity_x)) ## around 9% decrease to 4% fired eventually this way
            if tau == 1.0:
                return sparse_x.view(n, c, h, w)
            
            # print("--- tau", tau)
            tau_x = x * torch.FloatTensor([1. - tau]).to(device)
            # print("sum of x used", tau_x.sum())
            return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x


        if self.mode in ["hyper_col_nn", "hyper_col_nn_remap"]:
            transformed_x = self.transform(x) # use 1x1 conv to transform to [n, 1, h, w]
            transformed_x_exp = torch.exp(transformed_x)
            transformed_x_normed = transformed_x_exp / transformed_x_exp.sum((2, 3), keepdim=True) # softmax [n, 1, h, w]
            # build mask
            keep_top_num = max(int(self.topk * h * w), 1)
            _, index = torch.topk(transformed_x_normed.reshape(n, 1, h * w), keep_top_num, dim=2)
            mask = torch.zeros((n, 1, h * w)).to(device).scatter_(2, index, 1.0).view(n, 1, h, w) # [n, 1, h, w]
            # straight-through mask
            st_mask = (mask - transformed_x_normed).detach() + transformed_x_normed
            out = st_mask * x # [n, c, h, w]
            return out
            
            if self.mode == "hyper_col_nn_remap":
                non_zeros_idx = torch.where(mask == 1.0)
                out = out[non_zeros_idx[0], :, non_zeros_idx[2], non_zeros_idx[3]] # [n, c, i, i]
                out = out.reshape(n, int(self.topk * (self.resolution ** 2)) * self.ch)  # [n, int(self.topk * (self.resolution ** 2)) * self.ch]
                # concate position encoding



            return out
        
        elif self.mode == "hyper_col_nnlayer_concept_attention_proto_local":
            transformed_x = self.transform(x) # use 1x1 conv to transform to [n, 1, h, w]
            transformed_x_exp = torch.exp(transformed_x)
            transformed_x_normed = transformed_x_exp / transformed_x_exp.sum((2, 3), keepdim=True) # softmax [n, 1, h, w]
            # im2col to build local sparse mask
            unfolded_x = F.unfold(transformed_x, self.kernel) # n, L, k1*k2*1
            keep_top_num = max(int(self.topk * unfolded_x.shape[2]), 1)
            _, index = torch.topk(unfolded_x, keep_top_num, dim=2)
            mask = torch.zeros_like(unfolded_x).scatter_(2, index, 1) 
            mask = F.fold(mask, (h, w), self.kernel) # n, 1, h, w
            mask = torch.clamp(mask, min=0.0, max=1.0) # n, L, k1*k2*1
            mask = mask.view(n, 1, h * w).detach().to(device)

            return out



class LocalLinearModule(nn.Sequential):
    def __init__(self, indim, outdim, hidden_dim):
        super(LocalLinearModule, self).__init__(
            nn.Linear(indim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outdim)
        )

class LocalConvModule(nn.Module):
    def __init__(self, local_indim, local_topk_num, which_conv, in_ch, out_ch=1, k=1, p=0, bias=False):
        super(LocalConvModule, self).__init__()
        self.conv = which_conv(in_ch, out_ch, kernel_size=k, padding=p, bias=bias)
        self.local_topk_num = local_topk_num
        self.local_indim = local_indim

    def forward(self, x, device='cuda', top_vc=None, activate=True, select_index=-1):
        """
        x: [n, c, local_h, local_w] 
        output: [n, local_in_dim], [n, 1, h, w], [n, c, local_h, local_w]
        """
        n, c, local_h, local_w = x.shape
        transformed_x = self.conv(x)
        transformed_x_exp = torch.exp(transformed_x)
        transformed_x_normed = transformed_x_exp / transformed_x_exp.sum((2, 3), keepdim=True)
        # build mask
        if not activate:
            # used for non target block during testing
            assert self.training == False, "should activate this local Conv mask block during training (self.training is {} now).".format(self.training)
            # build fake zeros activations
            concat_out = torch.zeros((n, self.local_indim)).to(device)
            return concat_out, None, None
        elif top_vc:
            # !!! only used for testing
            print("1 vc mask !! ")
            if select_index == -1:
                _, index = torch.topk(transformed_x_normed.reshape(n, 1, local_h * local_w), 1, dim=2)
                print("select top 1 vc (out of {})".format(self.local_topk_num))
            else:
                _, index = torch.topk(transformed_x_normed.reshape(n, 1, local_h * local_w), self.local_topk_num, dim=2)
                print("before: index", index.shape)
                select_index = min(max(select_index, 0), self.local_topk_num-1)
                print(f"selected vc index {select_index} (totoal {self.local_topk_num})")
                index = index[:, :, select_index:select_index+1]
                print("after: index", index.shape)
            mask = torch.zeros((n, 1, local_h * local_w)).to(device).scatter_(2, index, 1.0).view(n, 1, local_h, local_w) # [n, 1, h, w]
            # straight-through mask
            st_mask = (mask - transformed_x_normed).detach() + transformed_x_normed
            out = st_mask * x # [n, c, local_h, local_w]
            
            # extract only the selected hypercolumn
            select_hc_index = torch.where(mask == torch.FloatTensor([1.0]).to(device))
            out_efficient = out[select_hc_index[0], :, select_hc_index[2], select_hc_index[3]].view(n, -1) # n, 1 * c, do a split local_topk_num if you want to see each vc
            
            # add zeros into out_efficient to make it n, 1 * c
            zumbe = torch.zeros((n, (self.local_topk_num - 1) * c)).to(device)
            out_efficient = torch.cat([out_efficient, zumbe], dim=1)

            concat_out = torch.cat([out_efficient, st_mask.view(n, -1)], dim=1) # n, local_in_dim
            print('postion encode')
            messup = torch.zeros_like(st_mask.view(n, -1)).to(device)
            messup[:,1] = 1
            concat_out = torch.cat([out_efficient, messup], dim=1)
            print(st_mask.view(n, -1).mean(0))
        else:
            # used when training, topk mask
            _, index = torch.topk(transformed_x_normed.reshape(n, 1, local_h * local_w), self.local_topk_num, dim=2)
            mask = torch.zeros((n, 1, local_h * local_w)).to(device).scatter_(2, index, 1.0).view(n, 1, local_h, local_w) # [n, 1, h, w]
            # straight-through mask
            st_mask = (mask - transformed_x_normed).detach() + transformed_x_normed
            out = st_mask * x # [n, c, local_h, local_w]
            
            # extract only the selected hypercolumn
            select_hc_index = torch.where(mask == torch.FloatTensor([1.0]).to(device))
            out_efficient = out[select_hc_index[0], :, select_hc_index[2], select_hc_index[3]].view(n, -1) # n, local_topk_num * c, do a split local_topk_num if you want to see each vc
            # concate with mask
            # print("out_efficient {}; st_mask.view(n, -1) {}".format(out_efficient.shape, st_mask.view(n, -1).shape))
            concat_out = torch.cat([out_efficient, st_mask.view(n, -1)], dim=1) # n, local_in_dim
            
            
        assert self.local_indim == concat_out.shape[1], f"output dim ({concat_out.shape[1]}) mismatches with the desired local_in_dim {self.local_indim}"

        return concat_out, st_mask, out


        


class Sparsify_hypercol_local_modular(nn.Module):
    def __init__(self, topk, ch, which_conv, resolution, local_reduce_factor=4, local_hidden_dim=-1, channel_reduce_factor=10):
        super(Sparsify_hypercol_local_modular, self).__init__()
        self.topk = topk  # percent of the top reponse to keep
        self.myid = "local_modular_hyper_col"
        self.local_reduce_factor = local_reduce_factor # each axis of h and w are devided by local_reduce_factor equally
        assert int(resolution / self.local_reduce_factor) == resolution / self.local_reduce_factor, f"local_reduce_factor should be a factor of resolution {local_reduce_factor}"

        # calculate statistics
        self.ch = ch
        self.channel_reduce_factor = channel_reduce_factor
        self.h = self.w = resolution 
        self.local_h = int(self.h / self.local_reduce_factor)
        self.local_w = int(self.w / self.local_reduce_factor)
        ## topk 
        self.local_topk_num = max(1, int(self.local_h * self.local_w * self.topk))
        ## num of local block
        self.num_block = self.local_reduce_factor ** 2
        ## nn arch one hidden layer with relu activation (see class LocalModule)
        self.local_indim = self.local_topk_num * self.ch + self.local_h * self.local_w
        print("local_indim {}, local_topk_num {}; ch {}, local_h {}, local_w {}".format(self.local_indim,self.local_topk_num, self.ch, self.local_h, self.local_w))
        self.local_hidden_dim = self.local_indim if local_hidden_dim == -1 else local_hidden_dim
        self.local_out_channel = (self.ch // self.channel_reduce_factor + 1)
        self.local_outdim = self.local_h * self.local_w * self.local_out_channel
        print(f"Define Local Modular Network: indim {self.local_indim} -- hidden {self.local_hidden_dim} -- outdim {self.local_outdim} -- out channel {self.local_out_channel}")

        # define modular network
        self.which_conv = which_conv
        self.recover_module = LocalLinearModule(self.local_indim, self.local_outdim, self.local_hidden_dim)
        self.list_of_sparsify_modules = nn.ModuleList([LocalConvModule(self.local_indim, self.local_topk_num, self.which_conv, self.ch, out_ch=1, k=1, p=0, bias=False) \
                                                        for i in range(self.num_block)])
        self.out_conv = which_conv(self.local_out_channel, self.ch, kernel_size=1, padding=0, bias=False)
        
        print("check1--------------------")
        
    def forward(self, x, test_top1_blockindex=None, select_index=-1, device='cuda'):
        """sparsify locally, and then recover local sparsified hypercolumn into full feature map by multiple modular neural network
        param: 
            - x: [n, c, h, w] full feature map
        return:
            - recovered_x : [n, c, h, w] recovered full feature map
        """
        # segment locally (grid manner)
        list_of_activation = []
        # print("inside x {}; local_reduce_factor {}".format(x.shape, self.local_reduce_factor))
        for split_h in torch.split(x, self.local_h, dim=2):
            for split_hw in torch.split(split_h, self.local_w, dim=3):
                # print("inside forward split_hw", split_hw.shape)
                list_of_activation.append(split_hw) # [n, c, local_h, local_w]
         
        if test_top1_blockindex:
            print("test_top1_blockindex {} in {}-{}".format(test_top1_blockindex, self.h, self.h))
            # during testing
            test_top1_block_index_ = test_top1_blockindex[0] * self.local_reduce_factor + test_top1_blockindex[1]

        # sparsify and recover
        list_of_post_recover = []
        for i, local_activation in enumerate(list_of_activation): # [n, c, local_h, local_w]
            # specify testing block
            if (not self.training) and test_top1_blockindex:
                if i == test_top1_block_index_:
                    print(f"block {i} is tested")
                    activate = True 
                    top_vc = True
                else:
                    print(f"block {i} is deactivated")
                    activate = False 
                    top_vc = False
                    
            else:
                activate = True
                top_vc = False
                
                
            local_activation, _, _ = self.list_of_sparsify_modules[i](local_activation, top_vc=top_vc, activate=activate, select_index=select_index, device=device) # sparsified and reshaped (n, local_in_dim)
            local_activation = self.recover_module(local_activation).view(-1, self.local_out_channel, self.local_h, self.local_w) # [n, local_out_channel, local_h, local_w]
            # print("local_activation after recover reshape", local_activation.shape)
            list_of_post_recover.append(local_activation)
        
        
        # concate the grid
        x = torch.cat(torch.cat(list_of_post_recover, dim=3).chunk(self.local_reduce_factor, dim=3), dim=2) # n, local_out_channel, h, w

        # increase channel back to normal channel 
        x = self.out_conv(x)

        return x





class DiffSelection(nn.Module):
    """Differentialble hypercolumn selection via MLP (1x1 conv)"""
    def __init__(self, topk, which_conv, ch, hidden_ch, device="cuda", gumbel_temperature=1):
        super(DiffSelection, self).__init__()
        self.conv1 = which_conv(ch, hidden_ch, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = which_conv(hidden_ch, 1, kernel_size=1, padding=0, bias=False)
        self.device = device
        # differentiable temperature
        self.gumbel_temperature = torch.nn.Parameter(torch.FloatTensor([gumbel_temperature]).to(device)) # 
        self.gumbel_temperature.requires_grad = True
        # topk
        self.topk_percent = topk 

    
    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv1(x)
        x = self.relu 
        x = self.conv2(x) 

        # build differential mask from gambel softmax
        mask = self.gumbel_softmax(x, self.gumbel_temperature)

        return x * mask, mask

    def gumbel_softmax(self, logits, temperature, eps=1e-20):
        """
        input: [*, h, w]
        return: [*, h, w] an one-hot mask
        """
        n, c, h, w = shape = logits.size()

        # sample gumbel noise
        U = torch.rand(shape).to(self.device)
        gumbel_noise = -Variable(torch.log(-torch.log(U + eps) + eps))

        # construct gumbel softmax
        y = logits + gumbel_noise
        y = F.softmax(y.view(n, c, h * w) / temperature, dim=2).view(n, c, h, w) # n, c, h, w

        # topk selection
        keep_top_num = max(int(self.topk_percent * h * w), 1)
        _, index = torch.topk(transformed_x_normed.view(n, 1, h * w), keep_top_num, dim=2)
        mask = torch.zeros((n, 1, h * w)).to(self.device).scatter_(2, index, 1.0).view(n, 1, h, w) # mask : [n, 1, h, w]
        # straight-through mask
        st_mask = (mask - y).detach() + y
        
        return st_mask


class Sparse_vc_combination(nn.Module):
    """select the visual concept and then use guassian interpolation to fill in the wholes"""
    def __init__(self, topk, which_conv, ch, resolution, class_num, hidden_ch=None, device="cuda", gumbel_temperature=1.0):
        super(Sparse_vc_combination, self).__init__()
        self.myid = 'combine_vc_sparse_bottleneck'
        self.device = device
        self.topk = topk
        self.class_num = class_num
        self.h = self.w = resolution
        self.keep_top_num = max(int(self.topk * self.h * self.w), 1)
        if not hidden_ch:
            hidden_ch = max(int(ch / 2), 1)
        
        self.select_module = DiffSelection(self.topk, which_conv, ch, hidden_ch, device=self.device, gumbel_temperature=gumbel_temperature)
        # recover module (using linear combination for every pixels)
        self.recover_matrix = nn.ModuleList([torch.nn.Parameter(torch.rand(self.keep_top_num, self.h * self.w)) for _ in range(self.class_num)]) # [k, h * w]
        self.recover_matrix.requires_grad = True 
        
    def forward(self, x, y):
        n, c, h, w = x.shape
        # sparsify
        x, mask = self.select_module(x)
        _, index = torch.where(mask == 1.0)
        x = torch.transpose(x[index[0], :, index[2], index[3]].view(n, self.keep_top_num, c), 1, 2) # [n, c, k]
        # x @ recover_matrix
        x = torch.matmul(x, self.recover_matrix) # [n, c, h * w]
        x = x.view(n, c, h, w) # [n, c, h, w]

        """
        left work to do, how to fix the relative position of k?
        """
        return x


def gumbel_top_softmax(logits, k, temperature=0.1, eps=1e-20, device='cuda'):
    """
    input: [*, h, w]
    return: [*, h, w] an one-hot mask
    """
    n, c, h, w = shape = logits.size()

    # sample gumbel noise
    U = torch.rand(shape).to(device)
    gumbel_noise = -Variable(torch.log(-torch.log(U + eps) + eps))

    # construct gumbel softmax
    y = logits + gumbel_noise
    y = F.softmax(y.view(n, c, h * w) / temperature, dim=2).view(n, c, h, w) # n, c, h, w

    # top selection
    _, index = torch.topk(y.view(n, c, h * w), k, dim=2)
    # print(index[2])
    mask = torch.zeros((n, c, h * w)).to(device).scatter_(2, index, 1.0).view(n, c, h, w) # mask : [n, c, h, w]
    print(mask[0].sum().item(), c)
    # straight-through mask
    st_mask = (mask - y).detach() + y
    
    return st_mask, y



class Sparse_vc_map_combination(nn.Module):
    def __init__(self, topk, which_conv, ch, resolution):
        super(Sparse_vc_map_combination, self).__init__()
        self.myid = "vc_map_combination"
        self.topk = topk
        self.resolution = resolution
        self.ch = ch
        self.topk_num = max(int(self.topk * (self.resolution ** 2)), 1)

        self.map_conv = which_conv(self.ch, self.topk_num, kernel_size=1, padding=0, bias=False)


    def forward(self, x, temp=0.1):
        # print("temp: ", temp)
        n, c, h, w = x.shape
        k = self.topk_num
        # print(f"n {n}, c {c}, h {h}, w {w}, k {k}")
        # mapping 
        mapping = self.map_conv(x) # n, k, h, w
        # selecting the representive vc
        st_masks = gumbel_top_softmax(mapping, 1, temp) # [n, k, h, w]

        # print(st_masks.sum())
        # print(f"st_masks {st_masks.shape}")
        x = (st_masks.unsqueeze(2) * x.unsqueeze(1)).sum((3, 4)) # [n, k, 1, h, w] * [n, 1, c, h, w] -> [n, k, c, h, w] -> sum((3,4)) : [n, k, c]
        # select 
        print(f"before x transpose {x[0]}")
        x = torch.transpose(x, 1, 2) # [n, c, k]
        print(f"x transpose {x[0]}")
        
        # bilinear intepolation based on mapping
        mapping = mapping.view(n, k, h * w) # [n, k, hw]
        # print(f"mapping {mapping.shape}")
        ## softmax normalization
        mapping = F.softmax(mapping, dim=1) # mapping become probability [n, k, hw]
        x = torch.matmul(x, mapping) # batched matrix maltiplation [n, c, k] @ [n, k, hw] -> [n, c, hw]
        # print(f"before view x {x.shape}")
        x = x.view(n, c, h, w) # [n, c, h, w]

        return x

###########################################################################
# implicit
class Sine(nn.Module):
    """for implicit representation"""
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

#### recover by conv
class conv_recover_block(nn.Module):
    def __init__(self, which_conv, upsample, inch, outch):
        super(conv_recover_block, self).__init__()
        self.conv = which_conv(inch, outch)
        self.upsample = upsample
        self.activation = Sine()

    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)
        return x
        


class conv_recover(nn.Module):
    def __init__(self, which_conv, upsample, topk_num, ch, resolution, start_reso=4, minimal_ch=16, class_condition=False, class_info_dim=0, 
                    inner_class_dim=1, mask_reconstruct=False):
        super(conv_recover, self).__init__()
        self.ch = ch # input channel
        self.interm_ch = max(1, self.ch // 2) # intermediate channel
        self.one_vc_ch = max(self.interm_ch // topk_num, minimal_ch) # on indivual
        self.resolution = resolution 
        self.start_reso = start_reso
        self.topk_num = topk_num
        self.class_condition = class_condition # whether recover takes class information to conditional generate
        self.class_info_dim = class_info_dim # conditioned on class info
        self.inner_class_dim = inner_class_dim 
        # build mask for each recover vc
        self.mask_reconstruct = mask_reconstruct

        end_rb, start_rb = int(torch.log2(torch.Tensor([resolution])).item()), int(torch.log2(torch.Tensor([start_reso])).item()) 
        num_con_blocks = end_rb - start_rb
        
        # specify conv channels used
        self.conv_chs = [self.one_vc_ch // 2**i for i in range(num_con_blocks, 0, -1)]
        self.conv_chs += [self.one_vc_ch]
        
        print(end_rb, start_rb, num_con_blocks, len(self.conv_chs))
        print(f"start_reso {start_reso}; self.conv_chs {self.conv_chs}")
        # linear recover module 
        if self.class_info_dim > 0:
            self.linear_indim = self.ch + self.resolution + self.resolution + self.inner_class_dim # c + w + h + share_dim
        else:
            self.linear_indim = self.ch + self.resolution + self.resolution
        self.linear_outdim = self.start_reso * self.start_reso * self.conv_chs[0]
        self.linear_hidden_dim = self.linear_outdim
        print(f"linear_indim {self.linear_indim}; linear_outdim {self.linear_outdim}")
        self.linear1 = nn.Linear(self.linear_indim, self.linear_hidden_dim)
        self.linear_activation = Sine()
        self.linear2 = nn.Linear(self.linear_hidden_dim, self.linear_outdim)

        # conv module 
        self.conv_blocks = []
        for i in range(num_con_blocks):
            self.conv_blocks.append(conv_recover_block(which_conv, upsample, self.conv_chs[i], self.conv_chs[i+1]))
        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        if self.mask_reconstruct:
            self.last_conv = which_conv(self.topk_num * (self.conv_chs[-1] - 1), self.ch)
        else:
            self.last_conv = which_conv(self.topk_num * self.conv_chs[-1], self.ch)
        
        if self.class_info_dim > 0:
            self.class_conditional_module = nn.Sequential(
                nn.Linear(self.class_info_dim, 10 * self.inner_class_dim),
                Sine(),
                nn.Linear(10 * self.inner_class_dim, self.inner_class_dim),
                Sine()
            )

    def forward(self, x, x_position, y_position, class_info=None):
        """x is [n, k, c]
        x_position: [n, k, w]
        y_position: [n, k, h]
        class_info: [n,]
        """
        n, k, c = x.shape
        if class_info is not None:
            # print("self.class_condition")
            class_info = self.class_conditional_module(class_info)
            class_info = class_info.unsqueeze(1).repeat(1, k, 1) # [n, k, share_dim]
            x = torch.cat([x, x_position, y_position, class_info], dim=2).reshape(n * k, -1) # [n * k, c + w + h + share_dim]
        else:
            x = torch.cat([x, x_position, y_position], dim=2).view(n * k, -1) # [n * k, c + w + h]
        x = self.linear1(x)
        x = self.linear_activation(x)
        x = self.linear2(x).view(-1, self.conv_chs[0], self.start_reso, self.start_reso) # [n * k, c + w + h]
        # print(f"x_inter_1 {x.shape}")

        # convs
        for block in self.conv_blocks:
            x = block(x) # n * k, one_ch, h, w
            # print(f"x_inter {x.shape}") 
        
        # construct mask 
        
        if self.mask_reconstruct:
            # print(f"previous x {x.shape}")
            mask_x = x[:, 0:1, :, :]
            x = x[:, 1:, :, :]
            # print(f"current x {x.shape}")
            mask_x = F.softmax(mask_x.view(n * k, 1, -1), dim=2).view(n * k, 1, self.resolution, self.resolution) # n * k, 1, h, w
            x = mask_x * x # n * k, one_ch-1, h, w
            # print(f"masked_x {x.shape}; k {k}; n {n}; self.one_vc_ch {self.one_vc_ch}")
            x = x.view(n, k * (self.conv_chs[-1] - 1), self.resolution, self.resolution)
            x = self.last_conv(x) # [n, c, h, w]
            mask_x = mask_x.view(n, k, self.resolution, self.resolution) # [n, k, h, w]
        else:
            # concat then and refine
            x = x.view(n, k * self.one_vc_ch, self.resolution, self.resolution)
            x = self.last_conv(x) # [n, c, h, w]
            mask_x = None
        return x, mask_x # [n, c, h, w]


# import functools
# which_conv = functools.partial(SNConv2d, kernel_size=3, padding=1,
#                                                 num_svs=1, num_itrs=1,
#                                                 eps=1e-12) 
# upsample = functools.partial(F.interpolate, scale_factor=2)

# ch, resolution = 756, 16
# topk = int(0.1 * resolution**2)

# model = conv_recover(which_conv, upsample, topk, ch, resolution)

# count = 0
# for module in model.modules():
#     count += sum([p.data.nelement() for p in module.parameters()])
# print(f"parameter count: {count}")

# x = torch.rand(10, ch + resolution + resolution)
# out = model(x)
# print(out.shape)


class spatial_implicit_comb(nn.Module):
    """for each of the vc, use implicit NN to generate k probability. Each vc would be a combination of HW hypercolumns
    input: x [n, c, h, w]
    output: x [n, k, c], 
    """
    def __init__(self, ch, topk_num, resolution, class_info_dim=0, inner_class_dim=1, sparse_weight=False, weight_sparsity=0.3):
        super(spatial_implicit_comb, self).__init__()
        self.topk_num = topk_num
        self.weight_sparsity = weight_sparsity
        self.sparse_weight = sparse_weight

        if class_info_dim > 0:
            self.input_dim = ch + resolution * 2 + inner_class_dim
        else:
            self.input_dim = ch + resolution * 2 
        self.hidden_dim1 = self.input_dim
        self.hidden_dim2 = 2 * self.hidden_dim1
        self.output_dim = topk_num
        self.class_info_dim = class_info_dim
        self.inner_class_dim = inner_class_dim

        self.implicit_select_module = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim1),
                Sine(), # using sine activation according to SIREN
                nn.Linear(self.hidden_dim1, self.hidden_dim2),
                Sine(),
                nn.Linear(self.hidden_dim2, self.hidden_dim2),
                Sine(),
                nn.Linear(self.hidden_dim2, self.output_dim)
            )
        
        if self.class_info_dim > 0:
            self.class_conditional_module = nn.Sequential(
                nn.Linear(self.class_info_dim, 10 * self.inner_class_dim),
                Sine(),
                nn.Linear(10 * self.inner_class_dim, self.inner_class_dim),
                Sine(),
            )
    def position(self, n, h, w, device='cuda'):
        """find position encoding of the reshape operation n*h*w"""
        # for x (i.e. len(x) = w)
        x_position = torch.cat([torch.eye(w) for _ in range(n * h)]).to(device) # n * h * w, w
        # for y (i.e len(y) = h)
        y_position_i = torch.cat([torch.zeros(w, h).scatter_(1, torch.LongTensor([[i] for _ in range(w)]), 1.) for i in range(h)]) # w * h, h
        y_position = torch.cat([y_position_i.clone() for _ in range(n)]).to(device) # n * h * w, h

        return x_position, y_position

    def forward(self, x, device='cuda', class_info=None):
        """x: [n, c, h, w]
        class_info: [n, dim] or None
        """
        n, c, h, w = x.shape
        # reshape to [n*h*w, c]
        x = torch.transpose(x.view(n, c, h*w), 1, 2).reshape(n*h*w, c) # n * h * w, c

        # preparing position encoding for implicit
        x_position, y_position = self.position(n, h, w, device=device) # [n * h * w, w], [n * h * w, h]
        
        if class_info is not None:
            # prepare class condition info
            # print(f"class_info {class_info.shape}")
            class_info = self.class_conditional_module(class_info) # n, inner_classdim
            class_info = class_info.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1).reshape(n * h * w, self.inner_class_dim)
            # class_info = class_info.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(n, h, w, self.class_info_dim).reshape(n * h * w, self.class_info_dim) # [n * h * w, share_dim]
            implicit_input = torch.cat([x, x_position, y_position, class_info], axis=1) # [n * h * w, (c + w + h + share_dim)]
        else:
            # print("class info is None")
            implicit_input = torch.cat([x, x_position, y_position], axis=1) # [n * h * w, (c + w + h)]
            # print(f"implicit_input {implicit_input.shape}")

        # calculate k probability for n * h * w data
        weight = self.implicit_select_module(implicit_input) # # n * h * w, k
        weight = torch.transpose(weight.reshape(n, h * w, self.topk_num), 1, 2) # n, k, h * w

        ## normalize weight by softmax
        weight = F.softmax(weight, dim=2)

        # orthogonoal weights
        weight_TT = torch.matmul(weight, torch.transpose(weight, 1, 2))
        weight_TT = ((weight_TT - torch.eye(self.topk_num).to(device))**2).mean()
        # print(f"weight !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {weight_TT.shape}")
        # sparse weight ?
        if self.sparse_weight:
            weight = sparse_1d_func(weight, self.weight_sparsity, device) # [n, k, h * w]
        # batched @ to get sparsed vc
        sparse_vc = torch.matmul(weight, x.view(n, h * w, c)) # [n, k, c]
        # location info of the sparsity
        vc_x_position = torch.matmul(weight, x_position.view(n, h * w, w)) # [n, k, w]
        vc_y_position = torch.matmul(weight, y_position.view(n, h * w, h)) # [n, k, h]

        return sparse_vc, vc_x_position, vc_y_position, weight_TT


class vc_interaction_sa_module(nn.Module):
    def __init__(self, input_ch):
        """vc lateral connection via selected vcs' self attention"""
        super(vc_interaction_sa_module, self).__init__()
        self.input_ch = input_ch
        self.attention_ch = max(input_ch // 8, 1)
        self.theta = nn.Linear(self.input_ch, self.attention_ch)
        self.phi = nn.Linear(self.input_ch, self.attention_ch)
        self.psi = nn.Linear(self.input_ch, self.attention_ch)
        self.back_to_input_ch1 = nn.Linear(self.attention_ch, max(self.input_ch // 2, 1))
        self.back_to_input_activation = nn.ReLU()
        self.back_to_input_ch2 = nn.Linear(max(self.input_ch // 2, 1), self.input_ch)

        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        """x: [n, k, c]
        """
        n, k, c = x.shape
        x1 = self.theta(x.view(-1, c)).view(n, k, -1) # n, k, atten_ch
        x2 = self.phi(x.view(-1, c)).view(n, k, -1) # n, k, atten_ch
        x3 = self.psi(x.view(-1, c)).view(n, k, -1) # n, k, atten_ch

        # atten self
        cov = torch.matmul(x2, torch.transpose(x1, 1, 2)) # n, k, n * k
        # softmax to probability
        cov = F.softmax(cov, dim=1)
        o = torch.matmul(cov, x3) # n, k, atten_ch
        
        # map it back
        o = self.back_to_input_ch1(o)
        o = self.back_to_input_activation(o)
        o = self.back_to_input_ch2(o)

        return x + self.gamma * o # [n, k, c]
        

def sparse_1d_func(x, topk, device):
    """sparsity for 1d"""
    n, k, c = x.shape 
    keep_top_num = max(int(topk * c), 1)
    _, index = torch.topk(x.abs(), keep_top_num, dim=2)
    mask = torch.zeros_like(x).scatter_(2, index, 1.).to(device)
    # print("mask percent: ", mask.mean().item())
    sparse_x = mask * x
    return sparse_x.view(n, k, c)


class Implicit_sparse_vc_recover(nn.Module):
    def __init__(self, topk, ch, resolution, y_share_dim=0, vc_go_sparse=False, spatial_implicit_comb_sparse_weight=False, mask_reconstruct=False):
        super(Implicit_sparse_vc_recover, self).__init__()
        self.myid = "implicit_sparse_vc_recover"
        self.topk = topk # percent to keep
        self.ch = ch
        self.resolution = resolution 
        self.topk_num = max(1, int(self.topk * resolution**2))
        self.y_share_dim = y_share_dim

        # sparsify module
        self.spatial_implicit_comb = spatial_implicit_comb(self.ch, self.topk_num, self.resolution, class_info_dim=y_share_dim, sparse_weight=spatial_implicit_comb_sparse_weight)
        self.vc_go_sparse = vc_go_sparse
        # vc interactions
        self.vc_interaction1 = vc_interaction_sa_module(self.ch)
        self.vc_interaction2 = vc_interaction_sa_module(self.ch)

        # use normal conv2d for implicit recover
        which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
        upsample = functools.partial(F.interpolate, scale_factor=2)
        print("ch", self.ch)
        self.conv_recover = conv_recover(which_conv, upsample, self.topk_num, self.ch, self.resolution, class_info_dim=y_share_dim, mask_reconstruct=mask_reconstruct)

    def forward(self, x, device="cuda", class_info=None, eval_vc_index=None):
        """x : [n, c, h, w]"""
        n, c, h, w = x.shape
        if self.y_share_dim > 0:
            assert class_info is not None, f"class_info is {class_info.shape} while y_share_dim = {self.y_share_dim}"
        else:
            assert class_info is None, f"class_info is {class_info.shape} while y_share_dim = {self.y_share_dim}"
        
        sparse_vc, vc_x_position, vc_y_position, weight_TT = self.spatial_implicit_comb(x, device=device, class_info=class_info) # [n, k, c], [n, k, w], [n, k, h]
        # if eval vc, maskout others

        if self.vc_go_sparse:
            sparse_vc = sparse_1d_func(sparse_vc, self.topk, device)
        
        if eval_vc_index is not None:
            print(f"eval_vc_index is {eval_vc_index}")
            print(f"sparse_vc {sparse_vc.shape} {sparse_vc[0]}")
            print("sparse vc correlation")
            cov = torch.matmul(sparse_vc, torch.transpose(sparse_vc, 1, 2))[22]
            print(cov.shape)
            print(cov)
            for i in range(len(cov)):
                print(cov[i])
            print(cov.shape)
            # plot vc cov
            plot_vc(cov, sparse_vc[22])
            assert eval_vc_index < sparse_vc.shape[1], f"eval_vc_index {eval_vc_index} is invalid since [n, k, c] is {sparse_vc.shape}"
            sparse_vc = eval_vc_mask(sparse_vc, eval_vc_index)
            vc_x_position = eval_vc_mask(vc_x_position, eval_vc_index)
            vc_y_position = eval_vc_mask(vc_y_position, eval_vc_index)
        # interaction between vcs
        x = self.vc_interaction1(sparse_vc) # [n, k, c]
        x = self.vc_interaction2(x) # [n, k, c]


        # recover
        x, mask_x = self.conv_recover(x, vc_x_position, vc_y_position, class_info=class_info) # [n, k, c], [n, k, w], [n, k, h] -> [n, c, h, w]

        assert x.shape == (n, c, h, w), f"x.shape {x.shape} != {(n, c, h, w)}"

        return x, weight_TT, mask_x


def eval_vc_mask(sparse_vc, index):
    """sparse_vc: [n, k, c]
    out: [n, k, c] but only index on dim is activated, others are zero masked out"""
    eval_vc_mask = torch.zeros_like(sparse_vc)
    eval_vc_mask[:, index, :] = 1.
    # eval_vc_mask[:, index+2, :] = 1.
    print(f"eval_vc_mask {eval_vc_mask.sum()}; index {index}")
    out = sparse_vc * eval_vc_mask
    print(f"out {out.mean()}")
    return out


def plot_vc(cov, sparse_vc):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    cov = cov.numpy()
    plt.imshow(cov, cmap='hot', interpolation='nearest')
    plt.savefig("/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/1percent/evals/hypercolumn_sparse_implicit_recover_vc_sparse_comb_weight_sparse_10percent/vc_layer1.png")
    plt.close()

    
    plt.clf()
    plt.gca().set_aspect('equal')
    sparse_vc = sparse_vc.numpy()
    # sns.heatmap(sparse_vc, annot=False,  linewidths=.5)
    plt.imshow(sparse_vc, cmap='hot', interpolation='nearest')
    plt.savefig("/lab_data/leelab/tianqinl/BigGAN-PyTorch/scripts/1percent/evals/hypercolumn_sparse_implicit_recover_vc_sparse_comb_weight_sparse_10percent/vc_activation_layer1.png")
    plt.close()

################
# March 18 2021#
################
from sobel import Sobel
class SparseGradient_HW(nn.Module):
    """implment sparse selection on sober gradient"""
    def __init__(self, topk, mode, lambda_locality=0.5, lambda_activation_l1_norm=1, topk_channel=0.1, lambda_sum_channels=1e-2):
        super(SparseGradient_HW, self).__init__()
        self.topk = topk 
        self.myid = "sparse_sobel_hw"
        self.sobel_operation = Sobel()
        self.mode = mode
        self.lambda_locality = lambda_locality
        self.lambda_activation_l1_norm = lambda_activation_l1_norm
        self.topk_channel = topk_channel
        self.lambda_sum_channels = lambda_sum_channels

    def forward(self, x, tau, device='cuda'):
        n, c, h, w = x.shape
        x_reshape = x.view(n, c, h * w)
        keep_top_num = max(int(self.topk * h * w), 1)

        # select topk by image gradient 
        grad_magnitude = self.sobel_operation(x) # n, c, h, w
        grad_magnitude_reshape = grad_magnitude.reshape(n, c, -1)

        _, index = torch.topk(grad_magnitude_reshape.abs(), keep_top_num, dim=2)
        mask = torch.zeros_like(grad_magnitude_reshape).scatter_(2, index, 1).to(device) # n, c, h * w
        # print("mask percent: ", mask.mean().item())
        
        if float(self.mode) == 1.0:
            sparse_x = mask * x_reshape
            return sparse_x.view(n, c, h, w), 0.
        elif float(self.mode) == 1.1:
            sparse_x = mask * x_reshape
            sparse_x = sparse_x.view(n, c, h, w)
            
            # apply l1 norm regularization in each channel to induce sparsity
            reg = grad_magnitude_reshape.abs().mean() * self.lambda_activation_l1_norm
            # regularize the gradient compactness in each channel
            x_coord_prob = grad_magnitude.sum(3).reshape(n * c, h) # [n * c, h]
            y_coord_prob = grad_magnitude.sum(2).reshape(n * c, w) # [n * c, w]
            # ## want the x/y mass more concentrated
            # print(f"grad_magnitude.sum((1, 2, 3)) {grad_magnitude.sum((1, 2, 3)).shape}")
            # print(f"grad_magnitude.sum((2, 3)) {grad_magnitude.sum((2, 3)).shape}")
            weighted_by_magnitude = (grad_magnitude.sum((2, 3)) / grad_magnitude.sum((1, 2, 3)).reshape(-1, 1)).reshape(n * c, )
            x_coord_entropy = Categorical(probs = x_coord_prob).entropy() * weighted_by_magnitude# weighted by the total magnitude of that channel
            y_coord_entropy = Categorical(probs = y_coord_prob).entropy() * weighted_by_magnitude
            regularize_locality = (x_coord_entropy.mean() + y_coord_entropy.mean()) * self.lambda_locality

            reg = reg + regularize_locality
            return sparse_x, reg
        elif float(self.mode) == 1.2:
            """sparse out the entire channel"""
            sparse_x = mask * x_reshape
            sparse_x = sparse_x.view(n, c, h, w) # n, c, h, w

            # apply l1 norm regularization in each channel to induce sparsity
            reg = grad_magnitude_reshape.abs().mean() * self.lambda_activation_l1_norm
            # regularize the gradient compactness in each channel
            x_coord_prob = grad_magnitude.sum(3).reshape(n * c, h) # [n * c, h]
            y_coord_prob = grad_magnitude.sum(2).reshape(n * c, w) # [n * c, w]
            # ## want the x/y mass more concentrated
            # print(f"grad_magnitude.sum((1, 2, 3)) {grad_magnitude.sum((1, 2, 3)).shape}")
            # print(f"grad_magnitude.sum((2, 3)) {grad_magnitude.sum((2, 3)).shape}")
            weighted_by_magnitude = (grad_magnitude.sum((2, 3)) / grad_magnitude.sum((1, 2, 3)).reshape(-1, 1)).reshape(n * c, )
            x_coord_entropy = Categorical(probs = x_coord_prob).entropy() * weighted_by_magnitude# weighted by the total magnitude of that channel
            y_coord_entropy = Categorical(probs = y_coord_prob).entropy() * weighted_by_magnitude
            regularize_locality = (x_coord_entropy.mean() + y_coord_entropy.mean()) * self.lambda_locality
            reg = reg + regularize_locality

            # only activate 20 % entire column
            sparse_x_channel = sparse_x.abs().sum((2, 3)) # n, c
            sparse_x_channel_prob = sparse_x_channel / sparse_x_channel.sum(1)[:, None] # n, c
            ## topk sparse_x_channel
            keep_top_num = max(int(self.topk_channel * c), 1)
            _, index = torch.topk(sparse_x_channel_prob, keep_top_num, dim=1)
            sparse_channel_mask = torch.zeros_like(sparse_x_channel_prob).scatter_(1, index, 1).to(device) # n, c
            
            # print(f"sparse_channel_mask[:, :, None, None] {sparse_channel_mask[:, :, None, None].shape}")
            # print(f"sparse_x_channel_prob[:, :, None, None] {sparse_x_channel_prob[:, :, None, None].shape}")
            # print(f"sparse_x {sparse_x.shape}")
            sparse_x = sparse_x * ( (sparse_channel_mask + sparse_x_channel_prob - sparse_x_channel_prob.detach())[:, :, None, None])

            return sparse_x, reg
        elif float(self.mode) == 1.3:
            """regularize the weights"""
            sparse_x = mask * x_reshape
            sparse_x = sparse_x.view(n, c, h, w) # n, c, h, w

            # apply l1 norm regularization in each channel to induce sparsity
            reg = grad_magnitude_reshape.abs().mean() * self.lambda_activation_l1_norm
            # regularize the gradient compactness in each channel
            x_coord_prob = grad_magnitude.sum(3).reshape(n * c, h) # [n * c, h]
            y_coord_prob = grad_magnitude.sum(2).reshape(n * c, w) # [n * c, w]
            # ## want the x/y mass more concentrated
            # print(f"grad_magnitude.sum((1, 2, 3)) {grad_magnitude.sum((1, 2, 3)).shape}")
            # print(f"grad_magnitude.sum((2, 3)) {grad_magnitude.sum((2, 3)).shape}")
            weighted_by_magnitude = (grad_magnitude.sum((2, 3)) / grad_magnitude.sum((1, 2, 3)).reshape(-1, 1)).reshape(n * c, )
            x_coord_entropy = Categorical(probs = x_coord_prob).entropy() * weighted_by_magnitude# weighted by the total magnitude of that channel
            y_coord_entropy = Categorical(probs = y_coord_prob).entropy() * weighted_by_magnitude
            regularize_locality = (x_coord_entropy.mean() + y_coord_entropy.mean()) * self.lambda_locality
            reg = reg + regularize_locality

            # only activate 20 % entire column
            sparse_x_channel = sparse_x.abs().sum((2, 3)) # n, c
            
            reg += sparse_x_channel.mean() * self.lambda_sum_channels

            return sparse_x, reg



### Mar 20
### revisit sparse hw

class NewSparseHW(nn.Module):
    """implment sparse selection on topk with regularizations"""
    def __init__(self, topk, mode, lambda_locality=0.5, lambda_activation_l1_norm=1, topk_channel=0.3, lambda_reg_map_coverage=1):
        super(NewSparseHW, self).__init__()
        self.topk = topk 
        self.myid = "regularized_sparse_hw"
        self.mode = mode
        self.lambda_locality = lambda_locality
        self.lambda_activation_l1_norm = lambda_activation_l1_norm
        self.topk_channel = topk_channel
        self.lambda_reg_map_coverage = lambda_reg_map_coverage

    def forward(self, x, tau, device='cuda'):
        n, c, h, w = x.shape
        x_reshape = x.view(n, c, h * w)
        keep_top_num = max(int(self.topk * h * w), 1)

        _, index = torch.topk(x_reshape.abs(), keep_top_num, dim=2)
        mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(device) # n, c, h * w
        # print("mask percent: ", mask.mean().item())
        
       
        if float(self.mode) == 1.0:
            """sparse out the entire channel"""

            sparse_x = mask * x_reshape
            sparse_x = sparse_x.view(n, c, h, w) # n, c, h, w
            x_shape_abs = x.abs() # n, c, h, w

            # only activate 20 % entire column
            sparse_x_channel = sparse_x.abs().sum((2, 3)) # n, c
            sparse_x_channel_prob = sparse_x_channel / sparse_x_channel.sum(1)[:, None] # n, c
            ## topk sparse_x_channel
            keep_top_num = max(int(self.topk_channel * c), 1)
            _, index = torch.topk(sparse_x_channel_prob, keep_top_num, dim=1)
            sparse_channel_mask = torch.zeros_like(sparse_x_channel_prob).scatter_(1, index, 1).to(device) # n, c
            
            # print(f"sparse_channel_mask[:, :, None, None] {sparse_channel_mask[:, :, None, None].shape}")
            # print(f"sparse_x_channel_prob[:, :, None, None] {sparse_x_channel_prob[:, :, None, None].shape}")
            # print(f"sparse_x {sparse_x.shape}")
            sparse_x = sparse_x * ( (sparse_channel_mask + sparse_x_channel_prob - sparse_x_channel_prob.detach())[:, :, None, None])

            ## complementary loss
            ## selected channel original x should spatially sum up to one
            # gather the select channel
            index = index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
            selected_channel_x = torch.gather(x_shape_abs, 1, index) # [n, keep_top_num, h, w]
            selected_channel_x = selected_channel_x / selected_channel_x.sum((2, 3), keepdim=True)
            reg = ((selected_channel_x.sum((1)) - torch.ones(n, h, w).to(device))**2).mean() * self.lambda_reg_map_coverage





            ######
            ###### Regularize
            ######
            # apply l1 norm regularization in each channel to induce sparsity
            reg += x_shape_abs.mean() * self.lambda_activation_l1_norm

            # regularize the gradient compactness in each channel
            x_coord_prob = x_shape_abs.sum(3).reshape(n * c, h) # [n * c, h]
            y_coord_prob = x_shape_abs.sum(2).reshape(n * c, w) # [n * c, w]
            # ## want the x/y mass more concentrated
            # print(f"grad_magnitude.sum((1, 2, 3)) {grad_magnitude.sum((1, 2, 3)).shape}")
            # print(f"grad_magnitude.sum((2, 3)) {grad_magnitude.sum((2, 3)).shape}")
            weighted_by_magnitude = (x_shape_abs.sum((2, 3)) / x_shape_abs.sum((1, 2, 3)).reshape(-1, 1)).reshape(n * c, )
            x_coord_entropy = Categorical(probs = x_coord_prob).entropy() * weighted_by_magnitude# weighted by the total magnitude of that channel
            y_coord_entropy = Categorical(probs = y_coord_prob).entropy() * weighted_by_magnitude
            regularize_locality = - (x_coord_entropy.mean() + y_coord_entropy.mean()) * self.lambda_locality # negative since topk activation tends to be together
            reg = reg + regularize_locality

            return sparse_x, reg
        else:
            raise NotImplementedError(f"mode {float(self.mode)} not implemented")

