import torch, pickle, pdb
from models.classify_model import ClassifyModelMNIST
from models.moe import MoEWrapper
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


NUM_EXPERTS = 1
K_VALUE = 1 

device = 'mps'
transform = transforms.ToTensor()
#transforms.Normalize((0.1307,), (0.3081,)) 
# Download the MNIST dataset and apply the transformation
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

D_mnist = train_dataset.data.unsqueeze(1)/255.0
L_mnist = train_dataset.targets 

print(f"D_mnist.shape: {str(D_mnist.shape)}")
print(f"L_mnist.shape: {str(L_mnist.shape)}")

## load squares data 
f = open('data/squares/squares_data_10nums.pkl', 'rb')
data = pickle.load(f); f.close()
D_squares = data['D'].unsqueeze(1)
L_squares = data['L'].squeeze(1) - 1 

print(f"D_squares.shape: {str(D_squares.shape)}")
print(f"L_squares.shape: {str(L_squares.shape)}")

#D_squares = torch.Tensor()
#L_squares = torch.Tensor()
D = torch.cat((D_mnist, D_squares), dim=0)
L = torch.cat((L_mnist, L_squares), dim=0)

## randomly permute the dataset 
rndidx = torch.randperm(D.shape[0])
D = D[rndidx, :, :, :]
L = L[rndidx]

split_int = int(0.8*len(D))
D_tr = D[:split_int].to(device)
D_te = D[split_int:].to(device)

L_tr = L[:split_int].to(device)
L_te = L[split_int:].to(device)

print(f"D_tr.shape: {str(D_tr.shape)}")
print(f"L_tr.shape: {str(L_tr.shape)}")
print(f"D_te.shape: {str(D_te.shape)}")
print(f"L_te.shape: {str(L_te.shape)}")

class ClassifyModelMOE(torch.nn.Module): 
    def __init__(self): 
        super(ClassifyModelMOE, self).__init__() 
        self.conv_base_model = ClassifyModelMNIST(h_only=True, use_convnet=True).to(device)

        self.moe_model = MoEWrapper(
            input_dim = 32 * 10 * 10,
            output_dim = 128,
            expert_list = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(32 * 10 * 10, 256, device=device), 
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, 128, device=device), 
                            torch.nn.ReLU(),
                        ) for i in range(NUM_EXPERTS) 
                    ]
                ), 
            K = K_VALUE,
            glu_on = False, 
            device = device,
            expert_choice = False, 
            output_type = 'sum',
        )
        self.sm_linear = torch.nn.Linear(128, 10, device=device)

    def forward(self, x): 
        x = self.conv_base_model(x)
        x = self.moe_model(x)
        x = self.sm_linear(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

model = ClassifyModelMNIST().to(device)
model.zero_grad()

"""
model = torch.nn.Sequential(
    torch.nn.Flatten(start_dim=1), 
    torch.nn.Linear(784, 256, device=device), 
    torch.nn.ReLU(), 
    torch.nn.Linear(256, 128, device=device), 
    torch.nn.ReLU(), 
    torch.nn.Linear(128, 10, device=device), 
    torch.nn.Softmax(dim=1),
)
"""
num_samples = 256
num_steps = int(200 * len(D_tr) / num_samples) # 5 epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()

for i in range(num_steps): 
    model.train() 
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
    y = torch.nn.functional.one_hot(y.long(), num_classes=10).squeeze(1).float()
    #print(y)
    #print(logits)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    
    if i != 0 and i % 100 == 0:
        print(f"step: {i}, loss: {loss.item()}")
        model.eval()
        logits = model(D_te)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == L_te).float().mean()
        print(f"---- [Eval] ---- step: {i}, acc: {acc.item()}")
