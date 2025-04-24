import torch, pickle, pdb
from models.classify_model import ClassifyModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformation
transform = transforms.ToTensor()
#transforms.Normalize((0.1307,), (0.3081,)) 
# Download the MNIST dataset and apply the transformation
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
#batch_size = 64
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Accessing a batch of data
#images, labels = next(iter(train_loader))
#print(f"Shape of images batch: {images.shape}") #torch.Size([64, 1, 28, 28])
#print(f"Shape of labels batch: {labels.shape}") #torch.Size([64])

D_mnist = train_dataset.data.unsqueeze(1)/255.0
L_cl_mnist = torch.zeros(len(D_mnist))

f = open('data/squares/squares_data.pkl', 'rb')
data = pickle.load(f)
f.close()
D_squares = data['D'].unsqueeze(1)
L_cl_squares = torch.ones(len(D_squares))

D = torch.cat((D_mnist, D_squares), dim=0)
L = torch.cat((L_cl_mnist, L_cl_squares), dim=0)

## randomly permute the dataset 
rndidx = torch.randperm(D.shape[0])
D = D[rndidx, :, :, :]
L = L[rndidx]

split_int = int(0.8*len(D))
D_tr = D[:split_int]
D_te = D[split_int:]

L_tr = L[:split_int]
L_te = L[split_int:]


print(f"D_tr.shape: {str(D_tr.shape)}")
print(f"L_tr.shape: {str(L_tr.shape)}")
print(f"D_te.shape: {str(D_te.shape)}")
print(f"L_te.shape: {str(L_te.shape)}")

model = ClassifyModel()
num_samples = 128 
num_steps = int(5 * len(D_tr) / num_samples) # 5 epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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