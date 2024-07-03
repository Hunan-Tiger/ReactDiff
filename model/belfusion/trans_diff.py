import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import torch
from torch import nn
from torch.nn import init
from timm.models.vision_transformer import Attention


DTYPES = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64
}

class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()

from .nn1 import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

def modulate(x, context):
    return x + context


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, activation=Activation.silu, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.act_layer = activation
        self.norm1 = normalization(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = normalization(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
                                 self.act_layer.get_act(),
                                 normalization(mlp_hidden_dim),
                                 nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
                                 )

        self.context_modulation = nn.Sequential(
            self.act_layer.get_act(),
            nn.Linear(hidden_size, 2 * hidden_size + 2, bias=True)
        )

    def forward(self, x, cond):
        modulation = self.context_modulation(cond)
        context1, context2, gate1, gate2 = modulation[..., :self.hidden_size], modulation[..., self.hidden_size: 2*self.hidden_size], modulation[..., -2], modulation[..., -1]
        x = x + gate1.unsqueeze(-1) * self.attn(modulate(self.norm1(x), context1))
        x = x + gate2.unsqueeze(-1) * self.mlp(modulate(self.norm2(x), context2))
        return x
  

  
class Trans_Diff(nn.Module):
    def __init__(self, depth=12, model_channels=128, num_heads=4, mlp_ratio=4.0, activation=Activation.silu, dtype='float32', **block_kwargs):
        super().__init__()
        self.num_features = model_channels
        self.activation = activation
        self.dtype = DTYPES[dtype]
        self.act = activation.get_act()
        self.norm_cond = normalization(model_channels)
        self.cond_layers = nn.Sequential(self.norm_cond, self.act, nn.Linear(model_channels, model_channels))

        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(model_channels, num_heads, mlp_ratio, **block_kwargs) for _ in range(depth)])
        self.norm = normalization(model_channels)

        self.init_weights()

    def init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    pass
        self.apply(_basic_init)
                


    def forward(self, x, t, cond):
        '''
        x: [B, N, D]
        t: [B]
        cond: [B, N, D]
        return [B, N, D]
        '''
        t = timestep_embedding(t, self.num_features, dtype=self.dtype) # [B] -> [B, D]
        t = t[:, None, :].expand(-1, x.size(1), -1) # [B, D] -> [B, N, D]
        cond = self.cond_layers(cond + t)

        for block in self.blocks:
            x = block(x, cond)  

        x = self.norm(x)
        return x
    

if __name__ == '__main__':
    model = Trans_Diff()
    x = torch.randn(2, 50, 128)
    cond = torch.randn(2, 50, 128)
    t = torch.tensor([1, 2])
    out = model(x, t, cond)
    print(out.shape)