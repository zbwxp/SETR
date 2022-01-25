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
class Vit_conv_hourglass(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=4,
                 expand_index=11,
                 down_conv_style="3x3",
                 **kwargs):
        super(Vit_conv_hourglass, self).__init__(**kwargs)
        self.shrink_index = shrink_index
        self.expand_index = expand_index

        embed_dim = kwargs['embed_dim']
        if down_conv_style == "3x3":
            self.down = nn.Conv2d(embed_dim, embed_dim,
                                  kernel_size=3, stride=2, padding=1)

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
                x = x.transpose(1, 2).reshape(n, c, h, w)
                x = self.down(x)
                x = x.reshape(n, c, hw // 4).transpose(2, 1)
            elif i == self.expand_index:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = x.reshape(n, c, hw * 4).transpose(2, 1)
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
