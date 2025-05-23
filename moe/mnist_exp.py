import argparse
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

USE_MOE = False 
NUM_EXPERTS = 5 
K_VALUE = 3 
GLU_ON = True
OUTPUT_TYPE = 'concat_sum' 

gpu = torch.device('mps') 

# -- replicate the MoE wrapper -- 
class SimpleSMRouter(nn.Module): 
    def __init__(
            self, 
            input_dim: int, 
            num_experts: int, 
        ): 
        super(SimpleSMRouter, self).__init__() 

        self.h_layer = nn.Linear(input_dim, num_experts, device=gpu) 
        self.router_act = nn.Softmax(dim=0) ## expert choice 

    def forward(self, input: torch.Tensor): 
        return self.router_act(self.h_layer(input))

class RouterWithGLU(nn.Module): 
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int, 
            num_experts: int, 
            glu_on: bool,
        ): 
        super(RouterWithGLU, self).__init__() 
        self.glu_on = glu_on 

        self.h_layer_1 = nn.Linear(input_dim, hidden_dim, device=gpu) 
        if glu_on: 
            self.h_layer_1_glu = nn.Linear(input_dim, hidden_dim, device=gpu) 
            self.h_glu_act = F.sigmoid 

        self.h_act = F.relu
        self.h_layer_norm = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False) 
                
        self.h_layer_2 = nn.Linear(hidden_dim, num_experts, device=gpu) 
        self.router_act = nn.Softmax(dim=0) ## expert choice 

    def forward(self, input: torch.Tensor): 
        h_1 = self.h_layer_1(input)

        if self.glu_on: 
            glu_mask = self.h_glu_act(self.h_layer_1_glu(input)) 
            h_1 = torch.mul(h_1, glu_mask) 

        h_1 = self.h_act(h_1) 
        h_1 = self.h_layer_norm(h_1) 
        return self.router_act(self.h_layer_2(h_1)) 

class MoEWrapper(nn.Module): 
    ## Expert Choice/Selection 
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            K: int, 
            expert_list: List[nn.Module], 
            glu_on: bool, 
            output_type: str = 'concat_sum' 
        ):
        super(MoEWrapper, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K 
        self.expert_list = expert_list 
        self.glu_on = glu_on 
        self.num_experts = len(expert_list) 
        self.output_type = output_type 
        router_hidden_dim = 256 
        self.router = RouterWithGLU(input_dim, router_hidden_dim, self.num_experts, glu_on=glu_on) 
        self.input_layer_norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=False)
        self.temp_output_layer_norm = torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False)
        if output_type == 'sum': 
            self.output_layer_norm = torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False)
        else: 
            self.output_layer_norm = torch.nn.LayerNorm(normalized_shape=output_dim * self.num_experts, elementwise_affine=False)
        #self.router = SimpleSMRouter(input_dim, self.num_experts) 
        self.renorm_sm = nn.Softmax(dim=0) 
        #self.rand_l = torch.rand((64, self.num_experts), device=gpu)
    
    def forward(self, input: torch.Tensor):
        #if len(self.expert_list) == 1: 
        #    return self.expert_list[0](input)
        input = self.input_layer_norm(input)
        l = self.router(input)
        #l = torch.rand((input.size()[0], self.num_experts), device=gpu)
        batch_K = math.ceil(self.K * 1.0 / self.num_experts * input.size()[0])
        ws, ib = torch.topk(l, batch_K, dim=0)
        #print('ws')
        #print(ws)
        #nws = ws
        nws = self.renorm_sm(ws) 
        #nws = ws 
        #nws = ws * 84.0 
        #print('nws')
        #print(nws)

        if self.output_type == 'sum': 
            output = torch.zeros((input.size()[0], self.output_dim), device=gpu) 
        else:
            output_list  = [torch.zeros(input.size()[0], self.output_dim, device=gpu) for i in range(self.num_experts)]
            output = torch.zeros((input.size()[0], self.output_dim * self.num_experts), device=gpu) 
            if self.output_type == 'concat_sum':
                temp_sum_output = torch.zeros((input.size()[0], self.output_dim), device=gpu) 
        
        for expert in range(self.num_experts): 
            selected_data = torch.index_select(input, 0, ib[:, expert]) 
            #selected_data = input
            selected_output = self.expert_list[expert](selected_data) 
            selected_output = torch.mul(selected_output, nws[:, expert].reshape((nws.size()[0],1))) 

            if self.output_type == 'sum': 
                output[ib[:, expert], :] += selected_output 
                #output += selected_output
            else:
                output_list[expert][ib[:, expert], :] = selected_output 
                if self.output_type == 'concat_sum':
                    temp_sum_output[ib[:, expert], :] += selected_output 
        
        if self.output_type != 'sum':
            if self.output_type == 'concat_sum': 
                temp_sum_output = self.temp_output_layer_norm(temp_sum_output)
                for i in range(self.num_experts): 
                    output_list[i] += temp_sum_output
            output = torch.concat(output_list, dim=1)
        output = self.output_layer_norm(output)
        return output

# -- make necessary changes in convnet code -- 
class Net(nn.Module):
    def __init__(self, use_moe: bool):
        super(Net, self).__init__()
        #self.dropout2 = nn.Dropout(0.5)

        self.model_type = 'cnn'

        if self.model_type == 'dnn': 
            self.moe_input_dim = 96 
            self.moe_output_dim = 64 
            ## attempt to use dnn arch 
            self.dnn_linear1 = nn.Linear(784, self.moe_input_dim, device=gpu)
            self.single_expert_instance = nn.Sequential(
                nn.Linear(self.moe_input_dim, self.moe_output_dim, device=gpu)
            )
        else: 
            self.moe_input_dim = 1152 
            self.moe_output_dim = 64 
            ## attempt to simplify conv arch 
            self.conv1 = nn.Conv2d(1, 8, 3, 1) 
            self.conv2 = nn.Conv2d(8, 8, 3, 1) 
            self.dropout1 = nn.Dropout(0.25)
            self.single_expert_instance = nn.Sequential(
                nn.Linear(self.moe_input_dim, self.moe_output_dim, device=gpu),
                nn.ReLU(), 
            )

        if use_moe: 
            self.moe_module_list = [
                #self.single_expert_instance
                #self.single_expert_instance for i in range(NUM_EXPERTS)
                copy.deepcopy(self.single_expert_instance) for i in range(NUM_EXPERTS)
                #nn.Sequential(
                #    nn.Linear(self.moe_input_dim, self.moe_output_dim, device=gpu),
                #    nn.ReLU(), 
                #) #for i in range(NUM_EXPERTS)
            ]
            self.moe_module = MoEWrapper(
                input_dim = self.moe_input_dim,
                output_dim = self.moe_output_dim,
                K = K_VALUE,
                expert_list = self.moe_module_list, 
                glu_on = GLU_ON, 
                output_type = OUTPUT_TYPE, 
            )
            if OUTPUT_TYPE != 'sum': 
                self.sm_linear = nn.Linear(self.moe_output_dim * NUM_EXPERTS, 10, device=gpu)
            else: 
                self.sm_linear = nn.Linear(self.moe_output_dim, 10, device=gpu)

        else: 
            self.moe_module = self.single_expert_instance
            self.sm_linear = nn.Linear(self.moe_output_dim, 10, device=gpu)

    def forward(self, x):

        if self.model_type == 'dnn': 
            x = torch.flatten(x, 1)
            x = F.relu(self.dnn_linear1(x))
            x = self.moe_module(x)
            x = self.sm_linear(x)
        else: 
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.moe_module(x)
            #x = self.dropout2(x)
            x = self.sm_linear(x)
        
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    #if use_cuda:
    #    device = torch.device("cuda")
    #elif use_mps:
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    device = gpu

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(use_moe=USE_MOE).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()