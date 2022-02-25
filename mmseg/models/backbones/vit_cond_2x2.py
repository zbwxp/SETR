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
class vit_cond_2x2(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=7,
                 expand_index=100,
                 num_queries=256,
                 use_softmax=False,
                 **kwargs):
        super(vit_cond_2x2, self).__init__(**kwargs)
        self.shrink_index = shrink_index
        self.expand_index = expand_index
        self.num_queries = num_queries
        reduction = 16
        expand_query = 1024
        self.num_queries = 256
        self.use_softmax = use_softmax
        if self.use_softmax:
            self.se = nn.Sequential(
                nn.Linear(expand_query, expand_query // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(expand_query // reduction, expand_query, bias=False),
                nn.Softmax()
            )
        else:
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
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                token = x.mean(-1)
                se = self.se(token)
                se = se.reshape(n, h, w)
                se = se[:, ::2, ::2]
                se = se.reshape(n, hw // 4)
                if self.use_softmax:
                    se = se * 100

                x = x.transpose(1, 2).reshape(n, c, h, w)
                down_x = x[:, :, ::2, ::2]
                x = down_x
                x = x.reshape(n, c, hw // 4).transpose(2, 1)

                x *= se[:, :, None]

            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
            if i == self.expand_index:
                break
        return tuple(outs)
