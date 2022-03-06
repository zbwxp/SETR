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
class vit_4x4_2step(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index_1=4,
                 shrink_index_2=7,
                 expand_index=100,
                 num_queries=256,
                 use_norm=False,
                 shrink="2x2",
                 **kwargs):
        super(vit_4x4_2step, self).__init__(**kwargs)
        self.shrink_index_1 = shrink_index_1
        self.shrink_index_2 = shrink_index_2
        self.expand_index = expand_index
        self.num_queries = num_queries
        self.use_norm = use_norm
        self.shrink = shrink
        if self.use_norm:
            self.shrink_norm = nn.LayerNorm(kwargs['embed_dim'])

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
            if i == self.shrink_index_1:
                x = blk(x, shrink=self.shrink)
            elif i == self.shrink_index_2:
                x = blk(x, shrink=self.shrink)
            else:
                x = blk(x)
            if i in self.out_indices:
                outs.append(x)
            if i == self.expand_index:
                break
        return tuple(outs)
