import torch
import torch.nn.functional as F
import numpy as np
import random
import copy


from model import ActorCritic
from experience_replay import ExperienceReplay


class Agent(object):
    def __init__(self, action_size, state_size,
                 fc_sizes=None,
                 actor_fc_sizes=[256,128,64],
                 critic_fc_sizes=[256,128,64],
                 gamma=0.99,
                 gae_tau=0.3,
                 tau=0.001,
                 lr=5e-4,
                 random_seed = 1234):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ac_target = ActorCritic(state_size,action_size,
                                     fc_sizes=fc_sizes,
                                     actor_fc_sizes=actor_fc_sizes,
                                     critic_fc_sizes=critic_fc_sizes).to(self.device)
        self.ac_local = ActorCritic(state_size,action_size,
                                    fc_sizes=fc_sizes,
                                    actor_fc_sizes=actor_fc_sizes,
                                    critic_fc_sizes=critic_fc_sizes).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.ac_local.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.ac_local.critic.parameters(), lr=lr)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        #some configuration constants
        self.gamma = gamma
        self.gae_tau = gae_tau
        self.tau = tau


    def act(self, state, add_noise=True):
        self.ac_local.eval()
        with torch.no_grad():
            action = self.ac_local.actor(state)
        self.ac_local.train()
        action = action.cpu().data.numpy()
        if(add_noise):
            action += self.noise.sample()
        return np.clip(action, -1,1)

    def learn(self, exp_replay, sample_train=10):
        for exp in range(0, sample_train):
            state, action, reward, next_state, done = exp_replay.sample()
            value = self.ac_target.critic(state, action)
            _, next_value = self.ac_target(next_state)

            q_target = reward + self.gamma*next_value*(1-done)
            q_expected = self.ac_local.critic(state,action)
            loss_critic = F.mse_loss(q_expected, q_target)

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_local.critic.parameters(), 0.8)
            self.optimizer_critic.step()

            action_pred = self.ac_local.actor(state)
            loss_actor = -self.ac_local.critic(state, action_pred).mean()

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_local.actor.parameters(), 0.8)
            self.optimizer_actor.step()

    def softUpdate(self):
        if(self.ac_target.fc_common != None):
            for target_param, local_param in zip(self.ac_target.fc_common.parameters(), self.ac_local.fc_common.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        for target_param, local_param in zip(self.ac_target.actor.parameters(), self.ac_local.actor.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        for target_param, local_param in zip(self.ac_target.critic.parameters(), self.ac_local.critic.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


'''
Noise class copied from ddpg_pendulum project
'''
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
