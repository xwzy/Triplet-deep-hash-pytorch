from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs


class DATASET(data.Dataset):
    pos_f1 = 'pos1'
    pos_f2 = 'pos2'
    neg_f = 'neg'
    train_file = 'train.pt'
    test_file = 'test.pt'
    train_images_number = 1000
    test_images_number = 20

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.prepare(self.train_images_number)

        if self.train:
            self.train_data = torch.load(
                os.path.join(root, self.train_file))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(root, self.test_file))

    def __getitem__(self, index):
        if self.train:
            imgx1, imgx2, imgy = self.train_data[0][index], self.train_data[1][index], self.train_data[2][index]
            imgx1 = imgx1.view(224, 224, 3).permute(2, 1, 0).float()
            imgx2 = imgx2.view(224, 224, 3).permute(2, 1, 0).float()
            imgy = imgy.view(224, 224, 3).permute(2, 1, 0).float()
            return imgx1, imgx2, imgy
        else:
            img, target = self.test_data[index], self.test_labels[index]
            img = torch.ByteTensor(img).view(224, 224, 3).permute(2, 1, 0)
            return img

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file))

    def prepare(self, number):
        from six.moves import urllib
        import gzip
        if self._check_exists():
            return
        training_set = (
            read_image_file(os.path.join(self.root, self.pos_f1), number),
            read_image_file(os.path.join(self.root, self.pos_f2), number),
            read_image_file(os.path.join(self.root, self.neg_f), number)
        )
        test_set = (
            read_image_file_test(os.path.join(self.root, self.pos_f1), os.path.join(self.root, self.neg_f),
                                 self.test_images_number),
            get_test_label(self.test_images_number)
        )

        with open(os.path.join(self.root, self.train_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Process Image Done!')


def read_image_file(path, number):
    files_path = os.listdir(path)
    print(files_path)
    length = len(files_path)
    times = number / length
    res = []
    for _ in range(int(times) + 1):
        for p in files_path:
            print(p)
            if p.startswith('.'):
                continue
            res.append(os.path.join(path, p))
    res = res[:number]
    print(len(res))
    images = []
    for a_image in res:
        img = Image.open(a_image).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = list(img.getdata())
        images.append(img)
    return torch.ByteTensor(images).view(-1, 224, 224, 3)


def read_image_file_test(pathPos, pathNeg, number):
    res = []
    files_path1 = os.listdir(pathPos)
    files_path1 = files_path1[:number]
    for p in files_path1:
        if p.startswith('.'):
            continue
        res.append(os.path.join(pathPos, p))
    files_path2 = os.listdir(pathNeg)
    files_path2 = files_path2[:number]
    for p in files_path2:
        if p.startswith('.'):
            continue
        res.append(os.path.join(pathNeg, p))
    images = []
    for a_image in res:
        img = Image.open(a_image).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = list(img.getdata())
        images.append(img)
    return torch.ByteTensor(images).view(-1, 224, 224, 3)


def get_test_label(number):
    a = [0 for _ in range(10)]
    b = [1 for _ in range(10)]
    return torch.LongTensor(a + b)
