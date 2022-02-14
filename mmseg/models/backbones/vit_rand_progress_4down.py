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
class Vit_rand_progress_4down(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=4,
                 expand_index=7,
                 num_queries=256,
                 **kwargs):
        super(Vit_rand_progress_4down, self).__init__(**kwargs)
        self.shrink_index = shrink_index
        self.expand_index = expand_index
        self.num_queries = num_queries
        self._iter = 0

    def forward(self, x):
        if self.training:
            self._iter += 1
            additonal_samples = int((max((80000-self._iter), 0)/80000)**2 * (1024-self.num_queries))
            num_samples = self.num_queries + additonal_samples
            print(num_samples)
        else:
            num_samples = self.num_queries

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
                idx = torch.randint(0, 1024, (num_samples,))
                x = x[:, idx]
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
            if i == self.expand_index:
                break
        outs.append(idx)
        return tuple(outs)
