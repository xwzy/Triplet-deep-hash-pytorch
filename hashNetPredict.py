# coding=utf-8

import torch.nn as nn
import math, os, time, random
import torch.utils.model_zoo as model_zoo
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class HashNetPredict(nn.Module):
    def __init__(self, in_channel=4096, hashLength=1024):
        super(HashNetPredict, self).__init__()
        self.fc = nn.Linear(in_channel, hashLength)
        self.sm = nn.Sigmoid()
        self.sma = nn.Softmax()
        # print(self.fc.weight.data)

    def forward(self, x1):
        x1 = self.sm(self.fc(x1))
        return x1
