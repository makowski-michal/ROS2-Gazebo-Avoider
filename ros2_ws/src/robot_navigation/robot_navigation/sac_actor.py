import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
import sac_utils as utils


class TanhTransform(pyd.transforms.Transform):
# takes any big number and squeezes it between -1 and 1, 1000 becomes 1.0, -500 becomes -1.0, 0 stays 0, we need this because robot can't go infinite speed
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p()) # reverse the squeezing

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh() # do the squeezing

    def _inverse(self, y):
        return self.atanh(y) # undo the squeezing

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x)) # math for correct probabilities


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
# creates a bell curve of possible actions, then squeezes them to [-1, 1], "usually go 0.5 speed, sometimes 0.3 or 0.7, but never 100"
    def __init__(self, loc, scale):
        self.loc = loc # center of the bell curve (favorite action)
        self.scale = scale # how spread out (how much to explore)
        self.base_dist = pyd.Normal(loc, scale) # make the bell curve
        transforms = [TanhTransform()] # squeeze it to [-1, 1]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
# the robot's brain that decides what to do, sees obstacle -> thinks -> says "go 0.3 speed, turn 0.5 angle"
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        self.log_std_bounds = log_std_bounds # limits on how random we can be
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.outputs = dict()
        self.apply(utils.weight_init) # start with random brain

    def forward(self, obs):
    # look at what's around, decide what to do
        mu, log_std = self.trunk(obs).chunk(2, dim=-1) # get average action and how random to be
        log_std = torch.tanh(log_std) # squeeze randomness
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1) # put in allowed range
        std = log_std.exp() # convert to actual randomness amount
        self.outputs["mu"] = mu
        self.outputs["std"] = std
        dist = SquashedNormal(mu, std) # create bell curve of actions
        return dist

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_actor/{k}_hist", v, step)
