# from .layers import DropPath, to_2tuple, trunc_normal_
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .decoder_attn import *

from .vit import VisionTransformer
from ..builder import BACKBONES

@BACKBONES.register_module()
class vit_decouple(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=4,
                 expand_index=100,
                 num_queries=256,
                 use_norm=False,
                 **kwargs):
        super(vit_decouple, self).__init__(**kwargs)
        self.shrink_index = shrink_index
        self.expand_index = expand_index
        self.num_queries = num_queries
        self.use_norm = use_norm


        dim = kwargs['embed_dim']
        num_heads = kwargs['num_heads']
        num_expand_layer = 3
        num_queries = 300
        decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
        self.decoder = TPN_Decoder(decoder_layer, num_expand_layer)
        self.q = nn.Embedding(num_queries, dim)


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
        attn = None
        for i, blk in enumerate(self.blocks):
            if i == self.shrink_index:
                bs = x.size()[0]
                x, attn = self.decoder(self.q.weight.repeat(bs, 1, 1).transpose(0, 1), x.transpose(0, 1))
                attn = attn.sigmoid()
                x = x.transpose(0, 1)
            else:
                x = blk(x)
            if i in self.out_indices:
                if attn is not None:
                    outs.append(torch.einsum("bqc,bql->blc", x, attn) / self.q.num_embeddings)
                else:
                    outs.append(x)

        return tuple(outs)
