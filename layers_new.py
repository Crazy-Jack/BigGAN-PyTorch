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
#    idea: convolutional selection              #
#################################################
"""perform hypercolumn sparsity since within each scope, it should be considered as 