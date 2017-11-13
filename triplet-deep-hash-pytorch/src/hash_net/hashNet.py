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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='hashNet')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.4)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=3)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# print(args.batch_size)

kwargs = {'num_worker': 1, 'pin_memory': True} if args.cuda else {}


class HashNet(nn.Module):
    def __init__(self, in_channel=1536, hashLength=1024):
        super(HashNet, self).__init__()
        self.fc = nn.Linear(in_channel, hashLength)
        self.sm = nn.Sigmoid()
        self.sma = nn.Softmax()
        self.initLinear()
        print(self.fc.weight.data)

    def forward(self, x1, x2, y):
        # x1 = self.sm(self.fc(self.sma(x1)))
        # x2 = self.sm(self.fc(self.sma(x2)))
        # y = self.sm(self.fc(self.sma(y)))

        # x1 = self.sm(self.fc(x1))
        # x2 = self.sm(self.fc(x2))
        # y = self.sm(self.fc(y))

        x1 = F.selu(self.fc(self.sma(x1)))
        x2 = F.selu(self.fc(self.sma(x2)))
        y = F.selu(self.fc(self.sma(y)))
        return x1, x2, y

    def initLinear(self):
        self.fc.weight.data.normal_(1.0, 0.33)
        self.fc.bias.data.fill_(0.1)


model = HashNet(in_channel=1536, hashLength=8192)

if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
pdist = nn.PairwiseDistance(2)


def train(epochs):
    model.train()
    pos1 = torch.load(open('../../feature/generated_dataset/pos1_fea.pt', 'rb'))
    pos2 = torch.load(open('../../feature/generated_dataset/pos2_fea.pt', 'rb'))
    neg = torch.load(open('../../feature/generated_dataset/neg_fea.pt', 'rb'))

    # print(pos1)
    # print(pos2)
    # print(neg)

    length = len(pos1)
    print(length)

    for epoch in range(epochs):
        for index in range(length):
            x1, x2, y = torch.FloatTensor(pos1[index]).view(1,-1), \
                        torch.FloatTensor(pos2[index]).view(1,-1), \
                        torch.FloatTensor(neg[index]).view(1,-1)
            # x1, x2, y = pos1[index],pos2[index],neg[index]
            # print('---------')
            # print(x1, x2, y)
            # print('---------')
            # x1 = x1.contiguous().view(1,3,28,28)
            # x2 = x2.contiguous().view(1,3,28,28)
            # y = y.contiguous().view(1,3,28,28)
            # print('---------')
            # print(batch_idx, x1, x2, y)
            # print('---------')

            # x1.type(torch.FloatTensor)
            # x2.type(torch.FloatTensor)
            # y.type(torch.FloatTensor)
            if args.cuda:
                x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
            x1, x2, y = Variable(x1), Variable(x2), Variable(y)
            optimizer.zero_grad()
            hash_x1, hash_x2, hash_y = model(x1, x2, y)
            loss1 = pdist(hash_x1, hash_x2)
            loss2 = pdist(hash_x1, hash_y)
            l = 10 - loss2 + loss1
            loss = F.relu(l)
            loss.backward()
            optimizer.step()
            if index % args.log_interval == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, index , 1000, 100.0 * index / 1000,
                #     loss.data[0]))
                print("==============================================")
                print("total loss=", loss.data[0])
                print("x1==",x1.data,'x2==',x2.data,'y==',y.data)
                print("hashx1=",hash_x1,"hashx2=",hash_x2,'hashy=',hash_y)
                print('loss1==', loss1.data[0][0], 'loss2==', loss2.data[0][0])
                # time.sleep(5)
        torch.save(model.state_dict(), '../../model/hashNetInceptionv4-epoch' + str(epoch) + '.pth')


if __name__ == '__main__':
    train(1)
