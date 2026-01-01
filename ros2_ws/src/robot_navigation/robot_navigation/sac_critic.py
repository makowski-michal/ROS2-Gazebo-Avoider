import torch
from torch import nn
import sac_utils as utils


class DoubleQCritic(nn.Module):
# two judges that score how good an action is, "if you go forward here, you'll get 50 points total", robot has two judges so we it wont get overconfident
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth) # first judge
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth) # 2nd judge
        self.outputs = dict()
        self.apply(utils.weight_init) # start with random brains

    def forward(self, obs, action):
    # judge this action: "ow many points will this get?
        assert obs.size(0) == action.size(0) # make sure we have same amount of both
        obs_action = torch.cat([obs, action], dim=-1) # combine what we see + what we want to do
        q1 = self.Q1(obs_action) # first judge's score
        q2 = self.Q2(obs_action) # 2nd judge's score
        self.outputs["q1"] = q1
        self.outputs["q2"] = q2
        return q1, q2

    def log(self, writer, step):
        for k, v in self.outputs.items():
            writer.add_histogram(f"train_critic/{k}_hist", v, step)
