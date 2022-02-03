# from .layers import DropPath, to_2tuple, trunc_normal_
import copy
import math

import torch
import torch.nn as nn
from torch import Tensor

from .vit import VisionTransformer
from ..builder import BACKBONES
from .mix_transformer import OverlapPatchEmbed_depth

class FilterTokenizer(nn.Module):
    """
    The Filter Tokenizer extracts visual tokens using point-wise convolutions.
    It takes input of size (HW, C) and outputs a tensor of size (L, D) where:
    - HW : height x width, which represents the number of pixels
    - C : number of input channels
    - L : number of tokens
    - D : number of token channels
    """

    def __init__(self, in_channels: int, token_channels: int, tokens: int) -> None:
        super(FilterTokenizer, self).__init__()

        self.tokens = tokens
        self.in_channels = in_channels
        self.token_channels = token_channels

        self.linear1 = nn.Linear(in_channels, tokens)
        self.linear2 = nn.Linear(in_channels, token_channels)

        self.cache1 = None
        self.cache2 = None
        self.token_cache = None

        # initialize weights
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Expected Input Dimensions: (N, HW, C), where:
        - N: batch size
        - HW: number of pixels
        - C: number of input feature map channels

        Expected Output Dimensions: (N, L, D), where:
        - L: number of tokens
        - D: number of token channels
        """

        a = self.linear1(x)  # of size (N, HW, L)
        self.cache1 = a
        a = a.softmax(dim=1)  # softmax for HW dimension, such that every group l features sum to 1
        self.cache2 = a
        a = torch.transpose(a, 1, 2)  # swap dimensions 1 and 2, of size (N, L, HW)
        a = a.matmul(x)  # of size (N, L, C)
        a = self.linear2(a)  # of size (N, L, D)

        self.token_cache = a
        return a


@BACKBONES.register_module()
class Vit_pos_depth(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 **kwargs):
        super(Vit_pos_depth, self).__init__(**kwargs)
        self.patch_embed1 = OverlapPatchEmbed_depth(patch_size=3, stride=1,
                                              in_chans=kwargs["embed_dim"],
                                              embed_dim=kwargs["embed_dim"])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x, H, W = self.patch_embed1(x)
        # stole cls_tokens impl from Phil Wang, thanks
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
