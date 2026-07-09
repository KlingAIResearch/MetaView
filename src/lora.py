import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.models.attention_processor import Attention
    
class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
    ):
        super().__init__()
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)