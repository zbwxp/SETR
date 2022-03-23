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
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

@BACKBONES.register_module()
class vit_decouple_v12(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=7,
                 expand_index=100,
                 num_queries=256,
                 use_norm=True,
                 **kwargs):
        super(vit_decouple_v12, self).__init__(**kwargs)
        self.shrink_index = shrink_index
        self.expand_index = expand_index
        self.num_queries = num_queries
        self.use_norm = use_norm


        dim = kwargs['embed_dim']
        num_heads = kwargs['num_heads']
        if self.use_norm:
            self.shrink_norm = nn.LayerNorm(dim)
        num_expand_layer = 3
        num_queries = 300
        decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
        self.decoder = TPN_Decoder(decoder_layer, num_expand_layer)
        self.q = nn.Embedding(num_queries, dim)
        self.register_buffer("init_once", torch.tensor(0))
        self.register_buffer("_iter", torch.tensor(0))
        self.zip_pos_embed = PositionalEncoding(kwargs['embed_dim'], 1025)


    def forward(self, x):
        self._iter += 1
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
                if self.use_norm:
                    x = self.shrink_norm(x)
                bs, num_token, ch = x.size()
                if self.init_once == 0:
                    self.init_once += 1
                    print("self.q init once!!!!!!!")
                    idx = torch.randint(1, num_token, (self.q.num_embeddings,))
                    init_param = x[:, idx]
                    self.q.weight = nn.Parameter(init_param[0])
                # q = self.q.weight + self.zip_pos_embed.pos_table.clone().detach()
                x, attn = self.decoder(self.q.weight.repeat(bs, 1, 1).transpose(0, 1), x.transpose(0, 1))
                x = x.transpose(0, 1)
                pos = self.zip_pos_embed.pos_table.clone().detach()
                pos = pos.repeat(bs, 1, 1)
                pos_combine = torch.einsum("bql,blc->bqc", attn, pos)
                x = x + pos_combine
            else:
                x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        # outs.append({"loss_similarity": loss_sim})
        return tuple(outs)
