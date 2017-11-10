# triplet-deep-hash-pytorch
pytorch implementation of "Fast Training of Triplet-based Deep Binary Embedding Networks".
http://arxiv.org/abs/1603.02844

This implementation currently only supports binary deep hash.

#DEMO
Deep hash for "A", "B".
![](a.jpeg)
![](aa.jpeg)
![](b.jpeg)
![](bb.jpeg)

# TODO
- [ ] Add multiclass support.
- [ ] Make code clean.
- [ ] Add more base networks.

# Usage
## Train
1. Create folder *pos1*, *pos2*, *neg*, and put two category pictures in 3 folder.
2. Run `extractFeatures.py` to extract feature for future use.
3. Run `hashNet.py` to train your triplet deep hash network.


## Test
1. Create folder *test*, and create *pos*, *neg* in *test* with pictures that you want to retrive.
2. Run `testQue.py` to query your picture set.
