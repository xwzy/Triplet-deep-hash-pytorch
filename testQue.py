from __future__ import print_function

import os
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from hashNetPredict import HashNetPredict
import collections
from displatImage import display9Image
# from displatImage import displayImage

hashModelPath = 'model/hashNetres50-epoch40.pth'
model = torchvision.models.resnet50(pretrained=True)

new_classifier = nn.Sequential(*list(model.children())[:-1])
model.classifier = new_classifier
model.eval()
hashNet = HashNetPredict(1000, 20000)
hashNet.load_state_dict(torch.load(open(hashModelPath, 'rb')))


def getHashCode(a_image='test/a.png'):
    img = Image.open(a_image).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = list(img.getdata())
    img = torch.ByteTensor(img).view(224, 224, 3)
    img = img.view(224, 224, 3).permute(2, 1, 0).float()
    f = model(Variable(img.view(1, 3, 224, 224), volatile=True))
    hashcode = hashNet(f)
    return hashcode.data


def getTop(path, dis, number=9):
    print(path)
    print(dis)
    top = []
    length = len(path)
    for index in tqdm(range(length)):
        if len(top) < number:
            top.append((dis[index], path[index]))
            top.sort(key=lambda d: d[0])
        else:
            if dis[index] > top[-1][0]:
                continue
            else:
                top = top[:-1]
                top.append((dis[index], path[index]))
                top.sort(key=lambda d: d[0])
        print(top)
    return top


if __name__ == '__main__':
    pos_path = []
    neg_path = []
    pos_dis = []
    neg_dis = []
    testNumber = 200
    q_path = 'test/b.png'
    q = getHashCode(q_path)
    pos_dir = os.listdir('test/pos')
    try:
        pos_dir.remove('.DS_Store')
    except:
        pass
    for i in tqdm(range(testNumber)):
        p = pos_dir[i]
        p_path = os.path.join('test/pos', p)
        pos_path.append(p_path)
        pos_dis.append(torch.dist(q, getHashCode(p_path), 2))
    ###########
    neg_dir = os.listdir('test/neg')
    try:
        neg_dir.remove('.DS_Store')
    except:
        pass
    for i in tqdm(range(testNumber)):
        p = neg_dir[i]
        p_path = os.path.join('test/neg', p)
        neg_path.append(p_path)
        neg_dis.append(torch.dist(q, getHashCode(p_path), 2))

    res = getTop(pos_path + neg_path, pos_dis + neg_dis)
    # displayImage(q_path)
    display9Image(res)


