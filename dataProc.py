from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json

class readData(data.Dataset):
    def __init__(self, datafolder, train = True, npoints = 2500):
        self.dataFolder = datafolder
        self.npoints = npoints # sample npoints for training

        self.codeFile = os.path.join(self.dataFolder, 'synsetoffset2category.txt')
        self.cnameDic = {}
        self.datapath = []
        with open(self.codeFile, 'r') as f:
            for line in f:
                cname, code = line.strip().split();
                self.cnameDic[cname] = code;
        self.classes = dict(zip(sorted(self.cnameDic), range(len(self.cnameDic))))
        for item in self.cnameDic:
            dir_point = os.path.join(self.dataFolder, self.cnameDic[item], 'points')
            dir_seg = os.path.join(self.dataFolder, self.cnameDic[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) #
                self.datapath.append((item, os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

    def __getitem__(self, index):
        fn = self.datapath[index]
        points = np.loadtxt(fn[1]).astype(np.float32)  #(n*3,1)
        pointlabels = np.loadtxt(fn[2]).astype(np.int64)

        #sample npoints
        choice = np.random.choice(len(pointlabels), self.npoints, replace=True)
        # resample
        points = points[choice, :]
        pointlabels = pointlabels[choice]
        points = torch.from_numpy(points)
        pointlabels = torch.from_numpy(pointlabels)
        return points, pointlabels

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    print('test myloader:\n')
    d = readData(datafolder = '../firstcharm/shapenetcore_partanno_segmentation_benchmark_v0')
    print(len(d))
    points, pointlabels = d[100];
    print(points.size(), points.type(), pointlabels.size(), pointlabels.type())
    trainDataLoader = torch.utils.data.DataLoader(d, batch_size=64,
                                                  shuffle=True, num_workers=4)

