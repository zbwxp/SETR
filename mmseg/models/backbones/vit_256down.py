# from .layers import DropPath, to_2tuple, trunc_normal_
import math

import numpy as np
import torch
import torch.nn as nn

from .vit import VisionTransformer
from ..builder import BACKBONES
from ..utils.transformer_nn import TransformerDecoderLayer_returnattn, TransformerDecoder_returnattn
import matplotlib.pyplot as plt

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
class vit_256down(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=7,
                 expand_index=100,
                 num_queries=256,
                 backward_pos=False,
                 init_q_embedding=False,
                 **kwargs):
        super(vit_256down, self).__init__(**kwargs)
        embed_dim = kwargs['embed_dim']
        self.shrink_index = shrink_index
        self.expand_index = expand_index
        self.num_queries = num_queries
        self.backward_pos = backward_pos
        self.init_q_embedding = init_q_embedding

        self.zip_pos_embed = PositionalEncoding(embed_dim // 4, 1024)
        self.expand_query_embed = nn.Embedding(num_queries, embed_dim)
        self.expand_query_embed.weight.data *= 10
        decoder_layer = TransformerDecoderLayer_returnattn(d_model=embed_dim, nhead=12, dim_feedforward=2048)
        # for name, param in decoder_layer.named_parameters():
        #     param.data *= 10
        self.down_decoder = TransformerDecoder_returnattn(decoder_layer, 3)
        self._iter = 0

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
                bs = x.size()[0]
                down_pos = self.zip_pos_embed.pos_table.clone().detach().repeat(bs, 1, 1)
                q_embedding = self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1)
                if self.init_q_embedding:
                    n, hw, c = x.shape
                    h = w = int(math.sqrt(hw))
                    trans_x = x.transpose(1, 2).reshape(n, c, h, w)[:, :, ::2, ::2]
                    trans_x = trans_x.reshape(n, c, hw // 4).permute(-1, 0, 1)
                    q_embedding += trans_x
                down_x, attns = self.down_decoder(
                    q_embedding,
                    x.transpose(0, 1))
                if self.backward_pos:
                    down_pos = torch.bmm(attns[-1], down_pos)
                else:
                    down_pos = torch.bmm(attns[-1], down_pos).detach()
                x = down_x.transpose(0, 1)
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
            if i == self.expand_index:
                break
        outs.append(down_pos)
        return tuple(outs)
