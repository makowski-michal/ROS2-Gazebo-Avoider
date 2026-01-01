import numpy as np
import torch
import torch.nn.functional as F
from statistics import mean
import sac_utils as utils
from sac_critic import DoubleQCritic as critic_model
from sac_actor import DiagGaussianActor as actor_model
from torch.utils.tensorboard import SummaryWriter
import os


class SAC(object):
# the teacher that coordinates everything: the robot, the judges, the training
    def __init__(
        self,
        state_dim, # how many numbers describe what robot sees (82)
        action_dim, # how many actions robot can do (2: speed and turn)
        device, # use gpu or normal computer
        max_action, # biggest action value allowed
        discount=0.99, # how much we care about future vs now
        init_temperature=0.1, # how random to be at start
        alpha_lr=3e-4, # learning speed for randomness
        alpha_betas=(0.9, 0.999),
        actor_lr=3e-4, # learning speed for robot
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1, # teach robot every N times
        critic_lr=3e-4, # learning speed for judges
        critic_betas=(0.9, 0.999),
        critic_tau=0.005, # how fast to update stable judge
        critic_target_update_frequency=2, # update stable judge every N times
        learnable_temperature=True, # auto-adjust randomness
        save_every=100, # save progress every N lessons
        load_model=False, # start from saved brain?
        save_directory="./models",
        model_name="sac_model",
        load_directory="./models",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = (-max_action, max_action)
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory

        os.makedirs(save_directory, exist_ok=True)

        self.train_metrics_dict = {
            "train_critic/loss": [],
            "train_actor/loss": [],
            "train_actor/entropy": [],
            "train_alpha/value": [],
            "train/batch_reward": []
        }

        self.critic = critic_model( # judge thats learning
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=512,
            hidden_depth=2,
        ).to(self.device)
        
        self.critic_target = critic_model( # stable judge for reference
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=512,
            hidden_depth=2,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # copy values

        self.actor = actor_model( # the robot's brain
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=512,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ).to(self.device)

        if load_model:
            self.load(filename=model_name, directory=load_directory) # load saved brain

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device) # how random to be
        self.log_alpha.requires_grad = True # we can learn this
        self.target_entropy = -action_dim # target randomness level

        self.actor_optimizer = torch.optim.Adam( # teaches robot
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )
        self.critic_optimizer = torch.optim.Adam( # teaches judges
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )
        self.log_alpha_optimizer = torch.optim.Adam( # adjusts randomness
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.critic_target.train()
        self.actor.train(True)
        self.critic.train(True)
        self.step = 0
        self.writer = SummaryWriter()

    def save(self, filename, directory):
    # save everything we learned to files
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        torch.save(self.log_alpha, f"{directory}/{filename}_log_alpha.pth")
        print(f"Model saved to {directory}/{filename}")

    def load(self, filename, directory):
    # load what we learned before
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth"))
        if os.path.exists(f"{directory}/{filename}_log_alpha.pth"):
            self.log_alpha = torch.load(f"{directory}/{filename}_log_alpha.pth")
        print(f"Loaded model from: {directory}/{filename}")

    def train(self, replay_buffer, iterations, batch_size):
    # teach robot and judges by looking at memories
        for _ in range(iterations):
            self.update(replay_buffer=replay_buffer, step=self.step, batch_size=batch_size)

        for key, value in self.train_metrics_dict.items():
            if len(value):
                self.writer.add_scalar(key, mean(value), self.step)
            self.train_metrics_dict[key] = []
        
        self.step += 1

        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    @property
    def alpha(self):
        return self.log_alpha.exp() # current randomness level

    def act(self, obs, sample=True):
    # robot decides what to do based on what it sees, sample=True means try random stuff (exploring), False means do best thing (testing)
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs) # get possible actions
        action = dist.sample() if sample else dist.mean # pick random or best
        action = action.clamp(*self.action_range) # keep in safe range
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, done, step):
    # teach judge to predict scores better, judge learns: my guess should match (points now + expected future points)
        with torch.no_grad(): # use stable judge for targets
            dist = self.actor(next_obs) 
            next_action = dist.rsample() # what would robot do next?
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action) # stable judge's opinion
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob # take safer opinion
            target_Q = reward + ((1 - done) * self.discount * target_V) # total = now + future

        current_Q1, current_Q2 = self.critic(obs, action) # learning judge's guess
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) # how wrong?
        
        self.train_metrics_dict["train_critic/loss"].append(critic_loss.item())
        self.writer.add_scalar("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad() # teach judge to be more accurate
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step):
    # teach robot to pick better actions, robot learns by asking judge "was this good?" and improving
        dist = self.actor(obs)
        action = dist.rsample() # try an action
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action) # ask judge

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean() # want high score but stay a bit random
        
        self.train_metrics_dict["train_actor/loss"].append(actor_loss.item())
        self.train_metrics_dict["train_actor/entropy"].append(-log_prob.mean().item())
        self.writer.add_scalar("train_actor/loss", actor_loss, step)
        self.writer.add_scalar("train_actor/entropy", -log_prob.mean(), step)

        self.actor_optimizer.zero_grad() # teach robot
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature: # adjust how random to be
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            self.train_metrics_dict["train_alpha/value"].append(self.alpha.item())
            self.writer.add_scalar("train_alpha/value", self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step, batch_size):
     # one teaching session: pick memories, teach judge, teach robot
        batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states = \
            replay_buffer.sample_batch(batch_size) # pick random memories

        state = torch.Tensor(batch_states).to(self.device) # convert to gpu format
        next_state = torch.Tensor(batch_next_states).to(self.device)
        action = torch.Tensor(batch_actions).to(self.device)
        reward = torch.Tensor(batch_rewards).to(self.device)
        done = torch.Tensor(batch_dones).to(self.device)
        
        self.train_metrics_dict["train/batch_reward"].append(batch_rewards.mean().item())
        self.writer.add_scalar("train/batch_reward", batch_rewards.mean(), step) # always teach judge

        self.update_critic(state, action, reward, next_state, done, step)

        if step % self.actor_update_frequency == 0: # teach robot every few times
            self.update_actor_and_alpha(state, step)

        if step % self.critic_target_update_frequency == 0: # slowly update stable judge
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
