import math

import torch
import torch.nn as nn

from .vit import VisionTransformer
from ..builder import BACKBONES


# from .layers import DropPath, to_2tuple, trunc_normal_

@BACKBONES.register_module()
class Vit_Downsample(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 down_conv_style="1x1",
                 ziper_index=7,
                 ziper_query=256,
                 **kwargs):
        super(Vit_Downsample, self).__init__(**kwargs)

        self.ziper_index = ziper_index
        self.num_queries = ziper_query
        embed_dim = kwargs['embed_dim']
        if down_conv_style == "3x3":
            self.down = nn.Conv2d(embed_dim, embed_dim,
                                  kernel_size=3, stride=2, padding=1)
        elif down_conv_style == "1x1":
            self.down = nn.Conv2d(embed_dim, embed_dim,
                                  kernel_size=1, stride=2)

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
            if i == self.ziper_index:
                x = x[:, 1:]
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)
                x = self.down(x)
                x = x.reshape(n, c, hw//4).transpose(2, 1)

            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
