import torch, pickle, pdb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.classify_model import ClassifyModelMNIST
from models.moe import MoEWrapper

device = 'mps'
K_VALUE = 4
NUM_EXPERTS = 5
# Define the transformation
transform = transforms.ToTensor()

# Download the MNIST dataset and apply the transformation
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

D_mnist = train_dataset.data.unsqueeze(1)/255.0 
L_mnist = train_dataset.targets 

print(f"D_mnist.shape: {str(D_mnist.shape)}")
print(f"L_mnist.shape: {str(L_mnist.shape)}")

f = open('data/squares/squares_data_10nums.pkl', 'rb')
data = pickle.load(f); f.close()
D_squares = data['D'].unsqueeze(1)
L_squares = data['L'].squeeze(1) - 1 

print(f"D_squares.shape: {str(D_squares.shape)}")
print(f"L_squares.shape: {str(L_squares.shape)}")

#D_squares=torch.Tensor()
#L_squares=torch.Tensor()
D = torch.cat((D_mnist, D_squares), dim=0).to(device)
L = torch.cat((L_mnist, L_squares), dim=0).to(device)

rndidx = torch.randperm(len(D))
D = D[rndidx]
L = L[rndidx]

D = (D - torch.mean(D)) / torch.std(D)

split_int = int(0.8*len(D))
Dtr = D[:split_int]
Dte = D[split_int+1:]
Ltr = L[:split_int]
Lte = L[split_int+1:]

print(f"Dtr.shape: {str(Dtr.shape)}")
print(f"Ltr.shape: {str(Ltr.shape)}")

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
                            torch.nn.Linear(32 * 10 * 10, 128, device=device), 
                            torch.nn.Tanh(),
                            #torch.nn.ReLU(),
                            torch.nn.Linear(128, 128, device=device), 
                            torch.nn.Tanh(),
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
        x = torch.nn.Softmax(dim=1)(x)
        return x

#model = ClassifyModelMNIST(h_only=False, use_convnet=True).to(device)
model = ClassifyModelMOE().to(device)

"""
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 64, kernel_size=[3, 3]),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=1),
    torch.nn.Conv2d(64, 32, kernel_size=[4, 4]),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=2),
    torch.nn.Flatten(1),
    torch.nn.Linear(32 * 10 * 10, 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128, 10),
    torch.nn.Softmax(dim=1),
).to(device)
"""
model.zero_grad()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

batch_size = 512

num_steps = int(200 * len(Dtr) / batch_size)

for i in range(num_steps): 
    model.train()
    b = torch.randperm(len(Dtr))[:batch_size]
    x = Dtr[b, :, :, :]
    y = Ltr[b] 
    logits = model(x)
    y = torch.nn.functional.one_hot(y.long(), num_classes=10).squeeze(1).float()
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    if i != 0 and i % 100 == 0:
        print(f"step: {i}, loss: {loss.item()}")
        model.eval()
        logits = model(Dte)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == Lte).float().mean()
        print(f"---- [Eval] ---- step: {i}, acc: {acc.item()}")
