from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataProc import readData
from models import Pointseg
import torch.nn.functional as F


#set basic cons_parameter
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--numPoints', type=int,  default=2500, help='numpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

opt = parser.parse_args()
print (opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if opt.cuda else "cpu")

blue = lambda x:'\033[94m' + x + '\033[0m'

trainDataset = readData(datafolder='../firstcharm/shapenetcore_partanno_segmentation_benchmark_v0', npoints=opt.numPoints);
assert trainDataset

trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))

testDataset = readData(datafolder='../firstcharm/shapenetcore_partanno_segmentation_benchmark_v0', train= False);
assert testDataset

TestDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

blue = lambda x:'\033[94m' + x + '\033[0m'


print(len(trainDataset), len(testDataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

segmentor = Pointseg(opt.numPoints, 4).to(device)

if opt.model != '':
    Pointseg.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(segmentor.parameters(), lr=0.01, momentum=0.9)
#classifier.cuda()
#loss = nn.NLLLoss()  #F.nll_loss()

num_batch = len(trainDataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(trainDataLoader, 0):
        points, target = data
        points, target = points.to(device), target.to(device)

       # print(points.size(0), target.size())


        points = points.transpose(2, 1)

        optimizer.zero_grad()
        output = segmentor(points)
        #target = target -1
        output = output.transpose(2, 1).contiguous().view(-1,4)
        target = target.view(-1, 1)[:, 0] - 1
        #print(output.size(), target.size())

        #output = output.transpose(2, 1).view(-1, 4)
        error = F.nll_loss(output, target)
        error.backward()
        optimizer.step()

        pred_choice = output.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, error.item(), correct.item() / float(opt.batchSize * opt.numPoints)))
        if i % 10 == 0:
            j, data = next(enumerate(TestDataLoader, 0))
            points, target = data
            points, target = points.to(device), target.to(device)
            points = points.transpose(2, 1)


            output = segmentor(points)
            output = output.transpose(2, 1).contiguous().view(-1,4)
            target = target.view(-1, 1)[:,0] - 1

            error = F.nll_loss(output, target)

            pred_choice = output.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), error.item(), correct.item()/float(opt.batchSize * 2500)))
    torch.save(Pointseg.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))