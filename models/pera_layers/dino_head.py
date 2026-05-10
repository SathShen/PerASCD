# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm



def weight_norm(module, name='weight'):
    """Applies weight normalization to a parameter in the given module.
    
    Args:
        module (nn.Module): The module containing the parameter to be normalized.
        name (str, optional): The name of the parameter to be normalized. Default: 'weight'
    """
    # Get the parameter
    param = getattr(module, name)
    
    # Calculate the l2 norm of the parameter
    norm = param.norm(2, dim=1, keepdim=True)
    
    # Create weight_v and weight_g
    weight_v = nn.Parameter(param)
    weight_g = nn.Parameter(norm)
    
    # Replace the original parameter with the normalized parameter
    del module._parameters[name]
    module.register_parameter(name + '_v', weight_v)
    module.register_parameter(name + '_g', weight_g)

    # Ensure the weight normalization hook is applied every forward operation
    module.register_forward_pre_hook(lambda _, inputs: setattr(module, name, weight_g * weight_v / 
                                                               weight_v.norm(2, dim=1, keepdim=True).expand_as(weight_v)))


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, mlp_bias=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        weight_norm(self.last_layer)
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if (x.dtype == torch.float16 or x.dtype == torch.bfloat16) else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
