from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

class Transnet(nn.Module):
    def __init__(self, numpoints, k):
        super(Transnet, self).__init__()
        self.k = k
        self.numpoints = numpoints
        self.MLP_MAX = nn.Sequential(
            nn.Conv2d(self.k, 64, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d((numpoints, 1))
        )



        self.FCN = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        batchsize = x.size()[0]
        #print(x.size())
        x = self.MLP_MAX(x)
        x = x.view(-1, 1024)
        x = self.FCN(x)
        ident = torch.from_numpy(np.eye(self.k).astype(np.float32).flatten()).repeat(batchsize, 1).to(torch.device("cuda:0"))
        x = x + ident
        x = x.view(-1, self.k, self.k)
        return x




class Pointseg(nn.Module):
    def __init__(self, numpoints, classnum):
        super(Pointseg, self).__init__()
        self.numpoints = numpoints
        self.classnum = classnum
        self.transform1 = Transnet(numpoints, 3)
        self.transform2 = Transnet(numpoints, 64)
        self.MLP = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
        )

        self.MLP_F = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
        )

        self.MLP_SEG = nn.Sequential(
            nn.Conv2d(1088, 512, 1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, self.classnum, 1),
            #nn.BatchNorm2d(1024),
        )

        self.pool = nn.MaxPool2d((numpoints, 1))

    def forward(self, x):
        batchsize = x.size()[0]
        x = torch.unsqueeze(x, -1)
        trans3 = self.transform1(x)
        x = torch.squeeze(x, -1)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans3)
        x = x.transpose(2, 1)
        x = torch.unsqueeze(x, -1)
        #print(x.size())
        x = self.MLP(x)
        trans64 = self.transform2(x)
        x = torch.squeeze(x, -1)

        x = x.transpose(2, 1)
        x = torch.bmm(x, trans64)
        x = x.transpose(2, 1)
        x = torch.unsqueeze(x, -1)
        div_feature = x
        #print(x.size())
        x = self.MLP_F(x)
        x = self.pool(x)
        #print('111111111111111')
        #print(x.size())
        #print(div_feature.size())
        x = x.repeat(1, 1, self.numpoints, 1)
        x = torch.cat([div_feature, x], 1)
        x = self.MLP_SEG(x)
        x = torch.squeeze(x, -1)
        return x


if __name__ == '__main__':
    '''
    sim_data = Variable(torch.rand(32,2500,3))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = (k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())
    '''

    sim_data = torch.rand(32, 2500, 3)
    sim_data = sim_data.transpose(2,1)
    pointseg = Pointseg(sim_data.size()[2], 5)
    x = pointseg(sim_data)
    print(x.size())

