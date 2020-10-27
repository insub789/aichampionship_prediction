import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(1, '/workspace/Deep SVDD/')
from base.base_net import BaseNet
#from fastai import *

class baeminNet(BaseNet):
    def __init__(self):
        super().__init__()
        dropout_rate = 0.3
        self.rep_dim = 128
        
        #will change in future. maybe tabnet
        
        self.dp1 = nn.Dropout(p = dropout_rate)
        self.bn1d1 = nn.BatchNorm1d(25)
        self.fc1 = nn.Linear(25, 64, bias = False)
        self.act1 = nn.LeakyReLU(0.1)
        self.bn2d1 = nn.BatchNorm1d(64)
        self.dp2 = nn.Dropout(p = dropout_rate)
        self.fc2 = nn.Linear(64, 128, bias = True)
        self.act2 = nn.LeakyReLU(0.1)
        self.bn3d1 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(p = dropout_rate)
        self.fc3 = nn.Linear(128, self.rep_dim, bias = True)
        self.act3 = nn.LeakyReLU(0.1)
        self.bn4d1 = nn.BatchNorm1d(self.rep_dim)
    

    def forward(self, x):
        assert not torch.isnan(x).any()
        x = x.float()
        x = self.bn1d1(x)
        x = self.dp1(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.bn2d1(x) 
        x = self.dp2(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn3d1(x)
        x = self.dp3(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.bn4d1(x)
        return x

class baeminNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()



