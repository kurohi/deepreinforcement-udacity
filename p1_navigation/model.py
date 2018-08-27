import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv_sizes=[], kernels=[], strides=[], fc_sizes=[64,64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        sequence_dict = OrderedDict();
        for i,c_size in enumerate(conv_sizes):
            if(i == 0):
                sequence_dict['conv{}'.format(i)] = nn.Conv2d(state_size,c_size,kernels[i],strides[i])
            else:
                sequence_dict['conv{}'.format(i)] = nn.Conv2d(conv_sizes[i-1],c_size,kernels[i],strides[i])
            sequence_dict['relu{}'.format(i)] = nn.ReLU()
            #maybe add a max pooling here?
        if(len(conv_sizes)==0):
            sequence_dict['input_layer'] = nn.Linear(state_size, fc_sizes[0])
            sequence_dict['input_relu'] = nn.ReLU()
        for i,fc_size in enumerate(fc_sizes):
            if(i == len(fc_sizes)-1):
                sequence_dict['logit'] = nn.Linear(fc_size, action_size)
            else:
                sequence_dict['fc{}'.format(i)] = nn.Linear(fc_size, fc_sizes[i+1])
                sequence_dict['fc_relu{}'.format(i)] = nn.ReLU()
        #sequence_dict['output'] = nn.Softmax(dim=1)
        self.model = nn.Sequential(sequence_dict)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model.forward(state)
