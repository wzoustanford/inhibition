import torch, pickle, pdb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.classify_model import ClassifyModelMNIST, GlobalInhibitionModelV1
from models.moe import MoEWrapper

device = 'cuda'
K_VALUE = 3
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
D = torch.cat((D_mnist, D_squares), dim=0)
L = torch.cat((L_mnist, L_squares), dim=0)

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

        moe_output_type = 'sum'
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
            glu_on = True, 
            device = device, 
            expert_choice = False, 
            output_type = moe_output_type, 
        )
        if moe_output_type != 'sum': 
            self.sm_linear = torch.nn.Linear(128 * NUM_EXPERTS, 10, device=device)
        else: 
            self.sm_linear = torch.nn.Linear(128, 10, device=device)
        self.activations = {}
        self.get_activation_list = [
            "moe_model.router.h_glu_act",
            "moe_model.router.h_act",
            "moe_model",
            "sm_linear",
        ]
        for name, modu in self.named_modules():
            if name in self.get_activation_list:
                modu.register_forward_hook(self.get_activation(name))
    
    def get_activation(self, name): 
        def hook(module, input, output):
            self.activations[name] = output.detach().to(device)
        return hook
    
    def forward(self, x, gim_input = None): 
        x = self.conv_base_model(x)
        x = self.moe_model(x) if gim_input is None else self.moe_model(x, gim_input)
        x = self.sm_linear(x)
        x = torch.nn.Softmax(dim=1)(x)
        return x

#model = ClassifyModelMNIST(h_only=False, use_convnet=True).to(device)
model = ClassifyModelMOE().to(device)

xt = Dtr[:10, :, :, :].to(device)
yt = model(xt)
model.activations['y_labels'] = torch.zeros((10, 10)).to(device)
model.activations['cross_entropy'] = torch.Tensor().to(device)
gim_model = GlobalInhibitionModelV1(model.activations, device).to(device)

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
gim_model.zero_grad()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
optimizer_gim = torch.optim.Adam(gim_model.parameters(), lr=0.00005)

batch_size = 128
test_batch_size = 5120
iter_start_training_gim = 1

num_steps = int(25 * len(Dtr) / batch_size)

for i in range(num_steps): 
    model.train()
    b = torch.randperm(len(Dtr))[:batch_size]
    x_train_batch = Dtr[b, :, :, :]
    y_train_batch = Ltr[b] 
    x_train_batch = x_train_batch.to(device)
    y_train_batch = y_train_batch.to(device)
    if i < iter_start_training_gim:
        logits = model(x_train_batch)
    else: 
        h_gim = gim_model(model.activations)
        logits = model(x_train_batch, h_gim)
    
    y_train_batch = torch.nn.functional.one_hot(y_train_batch.long(), num_classes=10).squeeze(1).float()
    loss_train_batch = torch.nn.functional.cross_entropy(logits, y_train_batch)
    model.activations['y_labels'] = y_train_batch.detach()
    model.activations['cross_entropy'] = loss_train_batch.tile((batch_size, 1)).detach()
    loss_train_batch.backward()
    optimizer.step()
    optimizer_gim.step()
    
    if (i + 1) % 1000 == 0:
        print(f"step: {i}, loss: {loss_train_batch.item()}")
        model.eval()
        bt = torch.randperm(len(Dte))[:test_batch_size]
        x = Dte[bt, :, :, :]
        y = Lte[bt]
        x = x.to(device)
        y = y.to(device)
        if i < 1:
            logits = model(x)
        else: 
            b_tr_sim = torch.randperm(len(Dtr))[:test_batch_size]
            x_tr_sim = Dtr[b_tr_sim, :, :, :].to(device)
            y_tr_sim = Ltr[b_tr_sim].to(device)
            logits_tr_sim = model(x_tr_sim)
            y_tr_sim = torch.nn.functional.one_hot(y_tr_sim.long(), num_classes=10).squeeze(1).float()
            loss_tr_sim = torch.nn.functional.cross_entropy(logits_tr_sim, y_tr_sim)
            model.activations['y_labels'] = y_tr_sim.detach()
            model.activations['cross_entropy'] = loss_tr_sim.tile((test_batch_size, 1)).detach()
            h_gim_test = gim_model(model.activations)
            logits = model(x, h_gim_test)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        print(f"---- [Eval] ---- step: {i}, acc: {acc.item()}")

        model.train()
        _ = model(x_train_batch) #fill the activations again with the previous training batch to continue training
        model.activations['y_labels'] = y_train_batch.detach()
        model.activations['cross_entropy'] = loss_train_batch.tile((batch_size, 1)).detach()

