# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:35:20 2020

@author: zhewei
"""
import torch.nn as nn

# build the network      
class net(nn.Module):
    def __init__(self, input_size):
        super(net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20 , 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)