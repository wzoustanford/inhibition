import torch, math, pdb
import torch.nn as nn 
import torch.nn.functional as F

class RouterWithDefinedModelAndGLU(nn.Module): 
    def __init__(
            self, 
            defined_model: nn.Module,
            num_experts: int,
            device: torch.device,
        ): 
        super(RouterWithDefinedModelAndGLU, self).__init__() 
        self.defined_model = defined_model
        self.router = SimpleRouterWithGLU(defined_model.h_output_dim, num_experts).to(device)

    def forward(self, input: torch.Tensor): 
        h = self.defined_model(input)
        return self.router(h)
        
class SimpleRouterWithGLU(nn.Module): 
    def __init__(
            self, 
            input_dim: int, 
            num_experts: int, 
        ): 
        super(SimpleRouterWithGLU, self).__init__() 
                
        self.h_layer = nn.Linear(input_dim, num_experts) 
        self.router_act = nn.Softmax(dim=1) 

    def forward(self, input: torch.Tensor):         #h_1 = self.h_layer_norm(h_1) 
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

        self.h_layer_1 = nn.Linear(input_dim, hidden_dim) 
        if glu_on: 
            self.h_layer_1_glu = nn.Linear(input_dim, hidden_dim) 
            self.h_glu_act = F.sigmoid 

        self.h_act = F.relu
        #self.h_layer_norm = torch.nn.LayerNorm(hidden_dim) 
                
        self.h_layer_2 = nn.Linear(hidden_dim, num_experts) 
        self.router_act = nn.Softmax(dim=1) ## expert choice 

    def forward(self, input: torch.Tensor): 
        h_1 = self.h_layer_1(input)

        if self.glu_on: 
            glu_mask = self.h_glu_act(self.h_layer_1_glu(input)) 
            h_1 = torch.mul(h_1, glu_mask) 

        h_1 = self.h_act(h_1) 
        #h_1 = self.h_layer_norm(h_1) 
        return self.router_act(self.h_layer_2(h_1)) 

class MoEWrapper(nn.Module): 
    ## Expert Choice/Selection 
    def __init__(
            self, 
            defined_router_model: nn.Module,
            K: int, 
            expert_list: nn.ModuleList, 
            output_dim: int,
            glu_on: bool, 
            device: torch.device,
        ):
        super(MoEWrapper, self).__init__() 
        self.output_dim = output_dim
        self.K = K 
        self.expert_list = expert_list 
        self.glu_on = glu_on 
        self.num_experts = len(expert_list) 
        #router_hidden_dim = -1
        self.router= RouterWithDefinedModelAndGLU(defined_router_model, self.num_experts, device=device)
        
        #self.output_layer_norm = torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False)
        self.renorm_sm = nn.Softmax(dim=1) 
        self.device = device
        #self.rand_l = torch.rand((64, self.num_experts))

    def forward(self, input: torch.Tensor):
        if len(self.expert_list) == 1: 
            return self.expert_list[0](input) 

        #input = self.input_layer_norm(input)
        l = self.router(input)
        
        ws, ib = torch.topk(l, self.K, dim=1)
        nws = self.renorm_sm(ws) 

        output = torch.zeros((input.size()[0], self.output_dim)).to(self.device) 
        for expert in range(self.num_experts): 
            filter = torch.sum(ib == expert, dim=1) > 0 
            if filter.sum() == 0: 
                continue
            selected_data = input[filter] 
            #selected_data = torch.index_select(input, 0, ib[:, expert]) 
            selected_output = self.expert_list[expert](selected_data) 
            selected_output = torch.mul(selected_output, nws[filter][ib[filter] == expert].reshape((-1,1))) 
            output[filter] = selected_output
            #output[ib[:, expert], :] += selected_output 
        #output = self.output_layer_norm(output)
        return output

class MoEWrapperExpertSelection(nn.Module): 
    ## Expert Choice/Selection 
    def __init__(
            self, 
            defined_model: nn.Module,
            K: int, 
            expert_list: nn.ModuleList, 
            glu_on: bool, 
            device: torch.device,
        ):
        super(MoEWrapper, self).__init__() 
        self.K = K 
        self.expert_list = expert_list 
        self.glu_on = glu_on 
        self.num_experts = len(expert_list) 
        router_hidden_dim = 256 
        self.router= RouterWithDefinedModelAndGLU(defined_model, router_hidden_dim, self.num_experts, glu_on=glu_on, device=device)
        
        #self.output_layer_norm = torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False)
        self.renorm_sm = nn.Softmax(dim=0) 
        #self.rand_l = torch.rand((64, self.num_experts))

    def forward(self, input: torch.Tensor):
        if len(self.expert_list) == 1: 
            return self.expert_list[0](input) 

        #input = self.input_layer_norm(input)
        l = self.router(input)
        
        batch_K = math.ceil(self.K * 1.0 / self.num_experts * input.size()[0])
        ws, ib = torch.topk(l, batch_K, dim=0)
        #print('ws')
        #print(ws)
        #nws = ws
        nws = self.renorm_sm(ws) 
        #nws = ws * 84.0 
        #print('nws')
        #print(nws)
        
        output = torch.zeros((input.size()[0], self.output_dim)) 
        for expert in range(self.num_experts): 
            selected_data = torch.index_select(input, 0, ib[:, expert]) 
            selected_output = self.expert_list[expert](selected_data) 
            selected_output = torch.mul(selected_output, nws[:, expert].reshape((nws.size()[0],1))) 
            output[ib[:, expert], :] += selected_output 
        #output = self.output_layer_norm(output)
        return output
