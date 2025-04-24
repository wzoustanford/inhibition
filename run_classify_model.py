import torch, pickle, pdb
from models.classify_model import ClassifyModelCATSDOGS, ConvNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
data = torch.Tensor()
labels = []

# Load the CIFAR-10 dataset
for i in range(1, 6):
    d = pickle.load(open(f'data/CIFAR/data_batch_{i}', 'rb'), encoding='bytes')
    data = torch.cat((data, torch.Tensor(d[b'data'])), dim=0)
    labels += d[b'labels']
labels = torch.Tensor(labels)
"""
device = 'mps'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# filter for cats and dogs 
# cats: label 3 
# dogs: label 5 
data = torch.permute(torch.Tensor(trainset.data), (0, 3, 1, 2))
labels = torch.Tensor(trainset.targets)
test_data = torch.permute(torch.Tensor(testset.data), (0, 3, 1, 2))
test_labels = torch.Tensor(testset.targets)

filter = torch.logical_or(torch.eq(labels, 3), torch.eq(labels, 5))
cd_data = data[filter]
cd_labels = labels[filter]
cd_labels[cd_labels == 3] = 0
cd_labels[cd_labels == 5] = 1

d = pickle.load(open(f'data/CIFAR/test_batch', 'rb'), encoding='bytes')
test_data = torch.Tensor(d[b'data'])
test_labels = torch.Tensor(d[b'labels'])
# filter for cats and dogs
filter = torch.logical_or(torch.eq(test_labels, 3), torch.eq(test_labels, 5))
cd_test_data = test_data[filter]
cd_test_labels = test_labels[filter]
cd_test_labels[cd_test_labels == 3] = 0
cd_test_labels[cd_test_labels == 5] = 1

print(f"cd_data.shape: {str(cd_data.shape)}")
D_tr = cd_data.view(-1, 3, 32, 32).to(device)
L_tr = cd_labels.to(device)
D_te = cd_test_data.view(-1, 3, 32, 32).to(device)
L_te = cd_test_labels.to(device)

model = ConvNet().to(device)

num_samples = 128 
num_steps = int(500 * len(D_tr) / num_samples) # 5 epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
optimizer.zero_grad()

for i in range(num_steps): 
    model.train()
    model.zero_grad()
    rndidx= torch.randperm(D_tr.shape[0])[:num_samples]
    x = D_tr[rndidx, :, :, :]
    y = L_tr[rndidx] 
    #print(f"x.shape: {str(x.shape)}")
    #print(f"y.shape: {str(y.shape)}")
    #x = x.unsqueeze(0)
    #y = y.unsqueeze(0)
    # print(f"x.shape: {str(x.shape)}")
    # print(f"y.shape: {str(y.shape)}")
    logits = model(x)
    y = torch.nn.functional.one_hot(y.long(), num_classes=2).squeeze(1).float()
    #print(y)
    #print(logits)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    if i != 0 and i % 50 == 0:
        print(f"step: {i}, loss: {loss.item()}")
        model.eval()
        logits = model(D_te)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == L_te).float().mean()
        print(f"---- [Eval] ---- step: {i}, acc: {acc.item()}")