import numpy as np


from model import ActorCritic
from experience_replay import ExperienceReplay

class Agent(object):
    def __init__(self, action_size, state_sizes,
                 fc_sizes=None,
                 actor_fc_sizes=[256,128,64],
                 critic_fc_sizes=[256,128,64],
                 gamma=0.99,
                 gae_tau=0.3,
                 tau=0.001,
                 lr=5e-4):

        self.ac_target = ActorCritic(state_size,action_size,
                                     fc_sizes=fc_sizes,
                                     actor_fc_sizes=actor_fc_sizes,
                                     critic_fc_sizes=critic_fc_sizes).to(device)
        self.ac_local = ActorCritic(state_size,action_size,
                                    fc_sizes=fc_sizes,
                                    actor_fc_sizes=actor_fc_sizes,
                                    critic_fc_sizes=critic_fc_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.ac_local.parameters(), lr=lr)

        #some configuration constants
        self.gamma = gamma
        self.gae_tau = gae_tau
        self.tau = tau


    def act(self, state):
        _, action, value = self.ac_target(state)
        return action, value

    def learn(self, exp_replay, sample_train=10):
        for exp in range(0, sample_train):
            state, reward, next_state, done = exp_replay.sample()
            log_action, action, value = self.ac_target(state)
            _, _, next_value = self.ac_target(next_state)

            future_reward = reward + gamma*next_value*done
            advantage = future_reward - value
            entropy = (log_action * action).mean()

            loss_actor = (-log_action * advantage).mean()
            loss_critic = F.mse_loss(future_reward, value)

            loss = loss_actor + loss_critic*0.5 - 0.0001*entropy

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.ac_local.parameters(), 0.8)
            self.optimizer.step()

    def softUpdate(self):
        for target_param, local_param in zip(self.ac_target.parameters(), self.ac_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
