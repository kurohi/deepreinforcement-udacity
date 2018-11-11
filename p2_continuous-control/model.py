import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, deque

'''
Implementation of a generalizes actor-critic model
Adding the possibility to create a common network for both actor and critic before separating the two
if needed.
To make it continuous, got some inspiration from https://github.com/ShangtongZhang/
Specifically, how the GaussianActorCritic implementation
'''



class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, fc_sizes=None, actor_fc_sizes=[256,128,64], critic_fc_sizes=[256,128,64]):
        super(ActorCritic, self).__init__()
        #if the size of the common layers is specify, then create them
        #if not then set it to None so that it flags that that the actor and critic networks are separated
        if(fc_sizes!=None and (len(fc_sizes)>=1)):
            sequence_dict = OrderedDict();
            sequence_dict['fc0'] = nn.Linear(state_size, fc_sizes[0])
            sequence_dict['fc_rrelu0'] = nn.RReLU()
            for i,fc_size in enumerate(fc_sizes):
                if(i == len(fc_sizes)-1):
                    break
                sequence_dict['fc{}'.format(i+1)] = nn.Linear(fc_size, fc_sizes[i+1])
                sequence_dict['fc_rrelu{}'.format(i+1)] = nn.RReLU()

            self.fc_common = nn.Sequential(sequence_dict)
        else:
            self.fc_common = None

        if(self.fc_common != None):
            self.actor = Actor(fc_sizes[-1], action_size, actor_fc_sizes)
            self.critic = Critic(fc_sizes[-1], action_size, critic_fc_sizes)
        else:
            self.actor = Actor(state_size, action_size, actor_fc_sizes)
            self.critic = Critic(state_size, action_size, critic_fc_sizes)

        #weight initialization using xavier initializer
        if(self.fc_common != None):
            self.fc_common.apply(ActorCritic.init_weights)
        self.critic.critic_first_layer.apply(ActorCritic.init_weights)
        self.actor.fc_actor.apply(ActorCritic.init_weights)
        self.critic.fc_critic.apply(ActorCritic.init_weights)

        self.batchnorm = nn.BatchNorm1d(100)


    #the forward function is also dependent of the existence of the common layer
    #if it is not present, then the input of both actor and critics are the state
    def forward(self, state):
        state = self.batchnorm(state)
        if(self.fc_common != None):
            common_res = self.fc_common(state)
        else:
            common_res = state

        action = self.actor(common_res)
        value = self.critic(common_res, action)
        return  action, value

    #xavier initializer
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, actor_fc_sizes=[256,128,64]):
        super(Actor, self).__init__()

        #declaration of the actor sequences
        sequence_dict_actor = OrderedDict()

        sequence_dict_actor['fc0'] = nn.Linear(state_size, actor_fc_sizes[0])
        sequence_dict_actor['fc_rrelu0'] = nn.RReLU()
        for i,actor_fc_size in enumerate(actor_fc_sizes):
            if(i == len(actor_fc_sizes)-1):
                break
            sequence_dict_actor['fc{}'.format(i+1)] = nn.Linear(actor_fc_size, actor_fc_sizes[i+1])
            sequence_dict_actor['fc_rrelu{}'.format(i+1)] = nn.RReLU()
        sequence_dict_actor['logit'] = nn.Linear(actor_fc_sizes[-1], action_size)
        self.fc_actor = nn.Sequential(sequence_dict_actor)
        self.tanh = nn.Tanh()


    def forward(self, common_res):
        actor_res = self.fc_actor(common_res)
        action = self.tanh(actor_res)
        #dist = torch.distributions.Normal(action, self.std)
        #action = dist.sample()
        #log_actor = dist.log_prob(action)
        #log_actor = torch.sum(log_actor, dim=1, keepdim=True)
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size, critic_fc_sizes=[256,128,64]):
        super(Critic, self).__init__()

        #declaration of critic sequences
        sequence_dict_critic = OrderedDict()

        self.critic_first_layer = nn.Linear(state_size, critic_fc_sizes[0])
        sequence_dict_critic['fc0'] = nn.Linear(critic_fc_sizes[0]+action_size, critic_fc_sizes[0])
        sequence_dict_critic['fc_rrelu0'] = nn.RReLU()
        for i,critic_fc_size in enumerate(critic_fc_sizes):
            if(i == len(critic_fc_sizes)-1):
                break
            sequence_dict_critic['fc{}'.format(i+1)] = nn.Linear(critic_fc_size, critic_fc_sizes[i+1])
            sequence_dict_critic['fc_rrelu{}'.format(i+1)] = nn.RReLU()
        sequence_dict_critic['logit'] = nn.Linear(critic_fc_sizes[-1], 1)
        self.fc_critic = nn.Sequential(sequence_dict_critic)

    def forward(self, common_res, action):
        common_res = self.critic_first_layer(common_res)
        common_res = nn.RReLU(common_res)
        common_res = torch.cat((common_res.lower, action), dim=1)
        value = self.fc_critic(common_res)
        return value
