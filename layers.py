''' Layers
    This file contains various layers for the BigGAN models.
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from sync_batchnorm import SynchronizedBatchNorm2d as SyncBN2d


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
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
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
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
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

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
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
    return: 
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
        
        elif self.mode == "hyper_col_nn_local":
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
    def __init__(self, local_topk_num, which_conv, in_ch, out_ch=1, k=1, p=0, bias=False):
        super(LocalConvModule, self).__init__()
        self.conv = which_conv(in_ch, out_ch, kernel_size=k, padding=p, bias=bias)
        self.local_topk_num = local_topk_num

    def forward(self, x, device='cuda'):
        """
        x: [n, c, local_h, local_w] 
        output: [n, local_in_dim], [n, 1, h, w], [n, c, local_h, local_w]
        """
        n, c, local_h, local_w = x.shape
        transformed_x = self.conv(x)
        transformed_x_exp = torch.exp(transformed_x)
        transformed_x_normed = transformed_x_exp / transformed_x_exp.sum((2, 3), keepdim=True)
        # build mask
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
        self.list_of_sparsify_modules = nn.ModuleList([LocalConvModule(self.local_topk_num, self.which_conv, self.ch, out_ch=1, k=1, p=0, bias=False) \
                                                        for i in range(self.num_block)])
        self.out_conv = which_conv(self.local_out_channel, self.ch, kernel_size=1, padding=0, bias=False)
        
        print("check1--------------------")
        
    def forward(self, x, tau, device='cuda'):
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
         
        # sparsify and recover
        list_of_post_recover = []
        for i, local_activation in enumerate(list_of_activation): # [n, c, local_h, local_w]
            local_activation, _, _ = self.list_of_sparsify_modules[i](local_activation) # sparsified and reshaped (n, local_in_dim)
            local_activation = self.recover_module(local_activation).view(-1, self.local_out_channel, self.local_h, self.local_w) # [n, local_out_channel, local_h, local_w]
            # print("local_activation after recover reshape", local_activation.shape)
            list_of_post_recover.append(local_activation)
        
        
        # concate the grid
        x = torch.cat(torch.cat(list_of_post_recover, dim=3).chunk(self.local_reduce_factor, dim=3), dim=2) # n, local_out_channel, h, w

        # increase channel back to normal channel 
        x = self.out_conv(x)

        return x









