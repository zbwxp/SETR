# from .layers import DropPath, to_2tuple, trunc_normal_
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .vit import VisionTransformer
from ..builder import BACKBONES

@BACKBONES.register_module()
class Vit_cond_rand_4down(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=4,
                 expand_index=7,
                 down_conv_style="3x3",
                 **kwargs):
        super(Vit_cond_rand_4down, self).__init__(**kwargs)
        self.shrink_index = shrink_index
        self.expand_index = expand_index

        reduction = 16
        expand_query = 1024
        self.num_queries = 256
        self.se = nn.Sequential(
            nn.Linear(expand_query, expand_query // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(expand_query // reduction, expand_query, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.blocks):
            if i == self.shrink_index:
                x = x[:, 1:]
                token = x.mean(-1)
                learned_idx = torch.topk(self.se(token), self.num_queries)[1]
                x = torch.stack([x_[id_] for x_, id_ in zip(x, learned_idx)])
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
            if i == self.expand_index:
                break
        outs.append(learned_idx)
        return tuple(outs)
