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
from torch.distributions import Categorical