import pdb
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
        moe_inh_config: dict={},
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
        K_VALUE = moe_inh_config['K']
        NUM_EXPERTS = moe_inh_config['num_experts']
        GLU_ON = moe_inh_config['use_glu']
        self.ffnet = MoEWrapper(
            input_dim = d_model,
            output_dim = d_model,
            expert_list = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs), 
                            nn.ReLU(),
                            nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs), 
                        ) for i in range(NUM_EXPERTS) 
                    ]
                ), 
            K = K_VALUE, 
            glu_on = GLU_ON, 
            device = device, 
            expert_choice = False, 
            output_type = "sum", 
        )
    """
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor: 
        return super().forward(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            tgt_is_causal,
            memory_is_causal,
        )
    """
    def _ff_block(self, x: Tensor) -> Tensor:
        s = x.shape
        x = x.reshape((-1, s[-1]))
        x = self.ffnet(x)
        x = self.dropout3(x)
        x = x.reshape(s)
        return x