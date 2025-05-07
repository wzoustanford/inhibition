import torch, math, pdb
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional

class SimpleRouterWithGLU(nn.Module): 
    def __init__(
            self, 
            input_dim: int, 
            num_experts: int, 
            expert_choice: bool = False, 
        ): 
        super(SimpleRouterWithGLU, self).__init__() 
                
        self.h_layer = nn.Linear(input_dim, num_experts) 
        if expert_choice: 
            self.router_act = nn.Softmax(dim=0) 
        else:
            self.router_act = nn.Softmax(dim=1) 

    def forward(self, input: torch.Tensor): 
        return self.router_act(self.h_layer(input)) 

class TwoLayerRouterWithGLU(nn.Module): 
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int, 
            num_experts: int, 
            glu_on: bool,
            expert_choice: bool = False, 
            additional_input_dim: int = 0, 
        ): 
        super(TwoLayerRouterWithGLU, self).__init__() 
        self.glu_on = glu_on 
        if additional_input_dim <= 0: 
            additional_input_dim = 0
        #self.h_layer_1 = nn.Linear(input_dim + additional_input_dim, hidden_dim) 
        self.h_layer_1 = nn.Linear(input_dim, hidden_dim) 
        if glu_on: 
            self.h_layer_1_glu = nn.Linear(input_dim, hidden_dim) 
            self.h_layer_1_glu_gim = nn.Linear(128, hidden_dim)
            self.h_glu_act = nn.Sigmoid() #F.sigmoid 

        self.h_act = nn.Tanh() #F.tanh #F.relu 
        self.h_layer_2 = nn.Linear(hidden_dim, num_experts) 

        if expert_choice: 
            self.router_act = nn.Softmax(dim=0) 
        else:
            self.router_act = nn.Softmax(dim=1) 


    def forward(self, input: torch.Tensor, gim_input = None): 
        h_1 = self.h_layer_1(input)
        
        if additional_input is not None: 
            assert(additional_input.shape[1] == self.additional_input_dim)
            input = torch.cat((input, additional_input), dim=1)

        if self.glu_on: 
            if gim_input is None: 
                glu_lin_summed = self.h_layer_1_glu(input)
            else: 
                glu_lin_summed = self.h_layer_1_glu(input) + self.h_layer_1_glu_gim(gim_input)
            glu_mask = self.h_glu_act(glu_lin_summed) 
            h_1 = torch.mul(h_1, glu_mask) 

        h_1 = self.h_act(h_1) 
        
        h_2 = self.h_layer_2(h_1)
        #if self.glu_on: 
        #    glu_mask = self.h_glu_act(self.h_layer_2_glu(h_1))
        #    h_2 = torch.mul(h_2, glu_mask)
        
        return self.router_act(h_2) 

class MoEWrapper(nn.Module): 
    ## Expert Choice/Selection 
    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            expert_list: nn.ModuleList, 
            K: int, 
            glu_on: bool, 
            device: torch.device,
            expert_choice: bool = False, 
            output_type: str = 'sum',
            additional_input_dim: int = 0, 
        ):
        super(MoEWrapper, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K = K 
        self.expert_list = expert_list 
        self.glu_on = glu_on 
        self.num_experts = len(expert_list) 
        router_hidden_dim = 128 
        self.router = TwoLayerRouterWithGLU(
            input_dim, 
            router_hidden_dim, 
            self.num_experts, 
            glu_on=glu_on, 
            expert_choice=expert_choice, 
            additional_input_dim=additional_input_dim
        ).to(device) 
        
        self.input_layer_norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=False)
        self.additional_input_layer_norm = torch.nn.LayerNorm(normalized_shape=self.router.additional_input_dim, elementwise_affine=False)
        self.output_layer_norm = torch.nn.LayerNorm(normalized_shape=output_dim, elementwise_affine=False)
        if expert_choice: 
            self.renorm_sm = nn.Softmax(dim=0) 
        else: 
            self.renorm_sm = nn.Softmax(dim=1) 
        self.device = device
        self.expert_choice = expert_choice
        self.output_type = output_type


    def forward(self, input: torch.Tensor, gim_input=None):
        if len(self.expert_list) == 1: 
            return self.expert_list[0](input) 
        
        #input = self.input_layer_norm(input)
        
        l = self.router(input) if gim_input is None else self.router(input, gim_input)
        
        if self.expert_choice: 
            batch_K = math.ceil(self.K * 1.0 / self.num_experts * input.size()[0])
            ws, ib = torch.topk(l, batch_K, dim=0)
        else:
            ws, ib = torch.topk(l, self.K, dim=1)
        
        nws = self.renorm_sm(ws) 

        if self.output_type == 'sum': 
            output = torch.zeros((input.size()[0], self.output_dim)).to(self.device) 
        else:
            output_list  = [torch.zeros(input.size()[0], self.output_dim, device=self.device) for i in range(self.num_experts)]
            output = torch.zeros((input.size()[0], self.output_dim * self.num_experts), device=self.device) 
            if self.output_type == 'concat_sum':
                temp_sum_output = torch.zeros((input.size()[0], self.output_dim), device=self.device) 
        
        for expert in range(self.num_experts): 
            if self.expert_choice: 
                selected_data = torch.index_select(input, 0, ib[:, expert]) 
                selected_output = self.expert_list[expert](selected_data) 
                selected_output = torch.mul(selected_output, nws[:, expert].reshape((nws.size()[0],1))) 
                if self.output_type == 'sum': 
                    output[ib[:, expert], :] += selected_output 
                else:
                    output_list[expert][ib[:, expert], :] = selected_output 
                    if self.output_type == 'concat_sum':
                        temp_sum_output[ib[:, expert], :] += selected_output
            else:
                filter = torch.sum(ib == expert, dim=1) > 0 
                if filter.sum() == 0: 
                    continue
                selected_data = input[filter] 
                #selected_data = torch.index_select(input, 0, ib[:, expert]) 
                selected_output = self.expert_list[expert](selected_data) 
                selected_output = torch.mul(selected_output, nws[filter][ib[filter] == expert].reshape((-1,1))) 
                if self.output_type == 'sum': 
                    output[filter] += selected_output 
                else:
                    output_list[expert][filter] = selected_output 
                    if self.output_type == 'concat_sum':
                        temp_sum_output[filter] += selected_output 

        if self.output_type != 'sum':
            if self.output_type == 'concat_sum': 
                #temp_sum_output = self.temp_output_layer_norm(temp_sum_output)
                for i in range(self.num_experts): 
                    output_list[i] += temp_sum_output
            output = torch.concat(output_list, dim=1)
        
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
