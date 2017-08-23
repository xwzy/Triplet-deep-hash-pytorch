# coding=utf-8

import torch.nn as nn
import math, os
import torch.utils.model_zoo as model_zoo
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from dataset import DATASET

__all__ = ['ResNet', 'resnet101']

model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=10)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
# torch.set_default_tensor_type('torch.ByteTensor')
# torch.set_default_tensor_type('torch.FloatTensor')
print('OK')
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args.batch_size)

kwargs = {'num_worker': 1, 'pin_memory': True} if args.cuda else {}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # num_classes = hash length
    def __init__(self, block, layers, num_classes=4096):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sm = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, y):
        print(torch.equal(x1.data, x2.data))
        print(torch.equal(x1.data, y.data))
        print(x1.data)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc(x2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return self.sm(x1), self.sm(x2), self.sm(y)


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


model = resnet101()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    mydata = DATASET(os.getcwd())
    # train_loader = torch.utils.data.DataLoader(mydata,batch_size=8)

    # for batch_idx, (x1, x2, y) in enumerate(train_loader):
    for batch_idx in range(100):
        x1, x2, y = mydata[batch_idx]
        # print('---------')
        # print(batch_idx, x1, x2, y)
        # print('---------')
        x1 = x1.contiguous().view(1,3,28,28)
        x2 = x2.contiguous().view(1,3,28,28)
        y = y.contiguous().view(1,3,28,28)
        print('---------')
        print(batch_idx, x1, x2, y)
        print('---------')

        # x1.type(torch.FloatTensor)
        # x2.type(torch.FloatTensor)
        # y.type(torch.FloatTensor)
        if args.cuda:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        x1, x2, y = Variable(x1), Variable(x2), Variable(y)
        optimizer.zero_grad()
        hash_x1, hash_x2, hash_y = model(x1, x2, y)
        loss1 = F.cross_entropy(hash_x1, hash_x2)
        loss2 = F.cross_entropy(hash_x1, hash_y)
        l = 1 - loss2 + loss1
        loss = torch.max(torch.cat([0, l]))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader),
                loss.data[0]))


        # def test():
        #     pass
        # model.eval()
        # test_loss = 0
        # correct = 0
        # for data, target in test_loader:
        #     if args.cuda:
        #         data, target = data.cuda(), target.cuda()
        #     data, target = Variable(data, volatile=True), Variable(target)
        #     output = model(data)
        #     test_loss += F.nll_loss(output, target, size_average=False).data[0]
        #     pred = output.data.max(1)[1]
        #     correct += pred.eq(target.data).cpu().sum()
        #
        # test_loss /= len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100.0*correct/len(test_loader.dataset)
        # ))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test()
