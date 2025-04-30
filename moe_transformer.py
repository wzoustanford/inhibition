from torch import nn, Tensor
from typing import Union, Callable
import torch.nn.functional as F
from moe import MoEWrapper

class MOETransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model, 
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        K_VALUE = 3 
        NUM_EXPERTS = 5 
        self.ffnet = MoEWrapper(
            input_dim = d_model,
            output_dim = 128,
            expert_list = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(d_model, dim_feedforward, device=device, bias=bias, **factory_kwarg), 
                            self.activation,
                            nn.Linear(dim_feedforward, d_model, device=device, bias=bias, **factory_kwarg), 
                        ) for i in range(NUM_EXPERTS) 
                    ]
                ), 
            K = K_VALUE, 
            glu_on = True, 
            device = device, 
            expert_choice = False, 
            output_type = "sum", 
        )
        
        def _ff_block(self, x: Tensor) -> Tensor:
            x = self.ffnet(x)
            return self.dropout3(x)