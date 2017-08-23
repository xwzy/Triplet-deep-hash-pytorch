import torch
import os
from PIL import Image
import torchvision, torch, os
from torch import nn
from torch.autograd import Variable
from dataset import DATASET
from torchvision import datasets
from tqdm import tqdm
import random

# def read_image_file(path, number):
#     files_path = os.listdir(path)
#     for i in files_path:
#         print(i)
#     length = len(files_path)
#     times = number / length
#     res = []
#     for _ in range(int(times)+1):
#         res += files_path
#     res = res[:number]
#     print(res)
#     print(len(res))
#
# read_image_file('/', 1000)


# img = Image.open('/Users/wzy/PycharmProjects/pytorchDeepHash/pos1/VmVydGlnbyBVcHJpZ2h0IDIgQlJLLnR0Zg==.png').convert('RGB')
#
# print(list(img.getdata()))
# print(type(img))


# def read_image_file(path, number):
#     files_path = os.listdir(path)
#     print(files_path)
#     length = len(files_path)
#     times = number / length
#     res = []
#     for _ in range(int(times) + 1):
#         for p in files_path:
#             print(p)
#             if p.startswith('.'):
#                 continue
#             res.append(os.path.join(path, p))
#     res = res[:number]
#     print(len(res))
#     images = []
#     print(res)
#     for a_image in res:
#         img = Image.open(a_image).convert('RGB')
#         img = list(img.getdata())
#         images.append(img)
#     print(images[0])
#     print(images[2])
#     print(images[5])
#     aaa = torch.ByteTensor(images).view(-1, 28, 28, 3).permute(0, 3, 1, 2)
#     print(aaa)
#     return aaa
#
# read_image_file('/Users/wzy/PycharmProjects/pytorchDeepHash/pos1', 10)

# with open('pos1_fea.pt', 'rb') as f:
#     tensor = torch.load(f)
#     print(tensor)


# print(int(random.random()*1000))
# print(int(random.random()*1000))
# print(int(random.random()*1000))
# print(int(random.random()*1000))
# print(int(random.random()*1000))


number = 9
a = [1,2,3,4,5,6,7,8,9]

print(a)

a.insert(3,8888)
print(a)
a.sort()
print(a)
a = a[:-1]
print(a)
print(len(a))