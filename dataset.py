# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 00:20:42 2020

@author: zhewei
"""

from torch.utils.data import Dataset

# create pytorch dataset
class interviewDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]