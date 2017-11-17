# Triplet-deep-hash-pytorch
Pytorch implementation of "Fast Training of Triplet-based Deep Binary Embedding Networks".
http://arxiv.org/abs/1603.02844


# Update 2017.11.13
Refactor this project.

Use code in https://github.com/kentsommer/keras-inceptionV4 to extract feature.


# DEMO
Deep hash for "A", "B".
![](https://raw.githubusercontent.com/xwzy/triplet-deep-hash-pytorch/master/demo_picture/a.jpeg)
![](https://raw.githubusercontent.com/xwzy/triplet-deep-hash-pytorch/master/demo_picture/aa.jpeg)
![](https://raw.githubusercontent.com/xwzy/triplet-deep-hash-pytorch/master/demo_picture/b.jpeg)
![](https://raw.githubusercontent.com/xwzy/triplet-deep-hash-pytorch/master/demo_picture/bb.jpeg)

# TODO
- [x] Add multiclass support.
- [x] Make code clean.
- [ ] Add more base networks.
- [ ] Add query code for new project.

# Usage
## Train
1. Put training pictures in `/train/[category-id]`, test pictures in `data/test`.
2. Run `src/extract_feature/batch_extarct_test.py` and `src/extract_feature/batch_extract_train.py` to extract feature for future use.
3. Run `src/hash_net/generate_random_dataset.py` to generate random training data.
4. Run `src/hash_net/hashNet.py` to train your triplet deep hash network.


~~## Test~~

~~1. Create folder *test*, and create *pos*, *neg* in *test* with pictures that you want to retrive.~~

~~2. Run `testQue.py` to query your picture set.~~
