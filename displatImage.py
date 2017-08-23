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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# img=mpimg.imread('test/a.png')
# plt.subplot(221);
# plt.imshow(img, cmap ='gray');
# plt.subplot(222);
# plt.imshow(img, cmap ='gray');
# plt.subplot(223);
# plt.imshow(img, cmap ='gray');
# plt.subplot(224);
# plt.imshow(img, cmap ='gray');
# plt.show()


def displayImage(img_path):
    img = mpimg.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.show()


def display9Image(top):
    img = mpimg.imread(top[0][1])
    plt.subplot(331)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[1][1])
    plt.subplot(332)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[2][1])
    plt.subplot(333)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[3][1])
    plt.subplot(334)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[4][1])
    plt.subplot(335)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[5][1])
    plt.subplot(336)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[6][1])
    plt.subplot(337)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[7][1])
    plt.subplot(338)
    plt.imshow(img, cmap='gray')

    img = mpimg.imread(top[8][1])
    plt.subplot(339)
    plt.imshow(img, cmap='gray')

    # plt.figure(2)
    plt.show()

# displayImage('test/a.png')