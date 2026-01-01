import numpy as np
import torch
from torch import nn
import random


def soft_update_params(net, target_net, tau):
    # slowly copy learned knowledge from one network to another - tau = how much to copy (0.005 = copy 0.5% each time)
    # formula: target = 0.5% *new + 99.5%* old
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def weight_init(m):
    # initialize neural network with random starting values - if all values started at zero, network wouldn't learn anything
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data) # special kind of random that helps learning
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0) # set all biases to zero


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)] # simple case: direct input to output
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)] # first layer + activation
        for i in range(hidden_depth - 1): # add middle layers
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim)) # final output layer
    if output_mod is not None:
        mods.append(output_mod) # optional final transformation
    trunk = nn.Sequential(*mods) # combine all layers into one
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy() # move to CPU, remove from computation tracking, convert
