import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque


class Agent(nn.Module):
    def __init__(self, env, fc_sizes=[16]):
        if(len(fc_sizes) == 0):
            raise AssertionError("At least 1 hidden layer size has to be defined")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(Agent,self).__init__()
        self.env = env
        self.s_size = env.observation_space.shape[0]
        self.a_size = env.action_space.shape[0]
        self.fc_sizes = fc_sizes
        self.fc_layers = []
        prev_size = self.s_size
        for cur_size in fc_sizes:
            self.fc_layers.append(nn.Linear(prev_size, cur_size).to(self.device))
            prev_size = cur_size
        self.fc_sizes.append(self.a_size)
        self.fc_layers.append(nn.Linear(prev_size, self.a_size).to(self.device))

    def getWeightsDim(self):
        weight_dims = []
        bias_dims = []
        weight_dims.append(self.s_size*self.fc_sizes[0])
        bias_dims.append(self.fc_sizes[0])
        for i in range(len(self.fc_sizes)-1):
            weight_dims.append(self.fc_sizes[i]*self.fc_sizes[i+1])
            bias_dims.append(self.fc_sizes[i+1])
        weight_dims.append(self.fc_sizes[-1]*self.a_size)
        bias_dims.append(self.a_size)
        return weight_dims, bias_dims


    def setWeights(self, weights, bias):
        if(len(weights)!=len(self.fc_layers)):
            raise AssertionError("The weights has to have the same amount of layers")
        if(len(bias)!=len(self.fc_layers)):
            raise AssertionError("The weights has to have the same amount of layers")
        prev_size = self.s_size
        for i in range(len(weights)):
            fc_w = torch.from_numpy(weights[i].reshape(prev_size,self.fc_sizes[i]))
            fc_b = torch.from_numpy(bias[i])
            self.fc_layers[i].weight.data.copy_(fc_w.view_as(self.fc_layers[i].weight.data))
            self.fc_layers[i].bias.data.copy_(fc_b.view_as(self.fc_layers[i].bias.data))
            prev_size = self.fc_sizes[i]

    def forward(self, x):
        for i in range(len(self.fc_layers)-1):
            x = F.relu(self.fc_layers[i](x))
        x = F.tanh(self.fc_layers[-1](x))
        return x.cpu().data

    def evaluate(self, weights, bias, gamma=1.0, max_t=5000):
        self.setWeights(weights,bias)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(self.device)
            action = self.forward(state)
            state,reward,done,_ = self.env.step(action.tolist())
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

