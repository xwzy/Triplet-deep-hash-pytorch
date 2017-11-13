import numpy as np
import torch
from tqdm import tqdm
import random
import os
from sklearn.preprocessing import normalize


def get2diff_below(number):
    """

    :param low: include low
    :param high: include high
    :return: two different number
    """
    first = random.randint(0, number - 1)
    offset = random.randint(1, number - 1)
    second = (first + offset) % number
    return first + 1, second + 1


feature_map = {}
for i in range(1, 31):
    feature_map[i] = np.load("../../feature/train/" + str(i) + ".npy")

pos1 = []
pos2 = []
neg = []

for i in tqdm(range(10000)):
    cate1, cate2 = get2diff_below(30)

    cate1_list = feature_map[cate1]
    cate2_list = feature_map[cate2]

    cate1_len = len(cate1_list)
    cate2_len = len(cate2_list)

    cate1_line = random.randint(1, cate1_len)
    cate2_line1, cate2_line2 = get2diff_below(cate2_len)

    cate1_line_tensor = cate1_list[cate1_line - 1]

    cate2_line1_tensor = cate2_list[cate2_line1 - 1]
    cate2_line2_tensor = cate2_list[cate2_line2 - 1]
    # print(cate2_line1_tensor.shape)

    pos1.append(cate2_line1_tensor /np.linalg.norm(cate2_line1_tensor))
    pos2.append(cate2_line2_tensor/np.linalg.norm(cate2_line2_tensor))
    neg.append(cate1_line_tensor/np.linalg.norm(cate1_line_tensor))

try:
    os.remove('../../feature/generated_dataset/pos1_fea.pt')
except:
    pass
try:
    os.remove('../../feature/generated_dataset/pos2_fea.pt')
except:
    pass
try:
    os.remove('../../feature/generated_dataset/neg_fea.pt')
except:
    pass

with open('../../feature/generated_dataset/pos1_fea.pt', 'wb') as f:
    torch.save(pos1, f)
with open('../../feature/generated_dataset/pos2_fea.pt', 'wb') as f:
    torch.save(pos2, f)
with open('../../feature/generated_dataset/neg_fea.pt', 'wb') as f:
    torch.save(neg, f)
