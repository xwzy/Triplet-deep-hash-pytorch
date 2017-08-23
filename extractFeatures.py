import torchvision, torch, os
from torch import nn
from torch.autograd import Variable
from dataset import DATASET
from tqdm import tqdm

mydata = DATASET(os.getcwd())

model = torchvision.models.resnet50(pretrained=True)

new_classifier = nn.Sequential(*list(model.children())[:-1])
model.classifier = new_classifier
print(model)
model.eval()


# input 3x224x224

# pos1 = model(Variable(mydata[0][0].view(1, 3, 224, 224)))
pos1 = []
for index in tqdm(range(1000)):
    f = model(Variable(mydata[index][0].view(1, 3, 224, 224), volatile=True))
    pos1.append(f.data)

print(pos1)
with open('feature/pos1_fea.pt', 'wb') as f:
    torch.save(pos1, f)



pos2 = []
for index in tqdm(range(1000)):
    f = model(Variable(mydata[index][1].view(1, 3, 224, 224), volatile=True))
    pos2.append(f.data)

print(pos2)
with open('feature/pos2_fea.pt', 'wb') as f:
    torch.save(pos2, f)


neg = []
for index in tqdm(range(1000)):
    f = model(Variable(mydata[index][2].view(1, 3, 224, 224), volatile=True))
    neg.append(f.data)

print(neg)
with open('feature/neg_fea.pt', 'wb') as f:
    torch.save(neg, f)
