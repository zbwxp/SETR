import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
# from .layers import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import build_norm_layer
import torch
from .vit_up_head import VisionTransformerUpHead, trunc_normal_
from ..utils.positional_encoding import PositionEmbeddingSine
from ..utils.transformer import Transformer

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


@HEADS.register_module()
class up_from_rand(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 expand_query=1024,
                 num_ziper_layer=3,
                 use_rand_idx=True,
                 num_expand_layer=3,
                 num_heads=12,
                 **kwargs):
        super(up_from_rand, self).__init__(**kwargs)

        self.use_idx = use_rand_idx
        dim = kwargs['embed_dim']
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
        self.up_decoder = nn.TransformerDecoder(decoder_layer, num_expand_layer)
        self.expand_query_embed = nn.Embedding(expand_query, dim)

        self.zip_pos_embed = PositionalEncoding(kwargs['embed_dim'], 1024)

        if self.in_channels != dim:
            self.input_proj = nn.Linear(self.in_channels, dim)
            trunc_normal_(self.input_proj.weight, std=.02)
        else:
            self.input_proj = nn.Sequential()


    def forward(self, x):
        idx = x[-1]
        x = self._transform_inputs(x)
        x = self.input_proj(x)
        bs = x.size()[0]
        if x.dim() == 3:
            if x.shape[1] % 32 != 0:
                x = x[:, 1:]
        # pick rand token
        if self.use_idx:
            if len(idx.size()) == 1:
                x += self.zip_pos_embed.pos_table.clone().detach()[:, idx]
            else:
                pos = self.zip_pos_embed.pos_table.clone().detach()
                x += torch.stack([p_[id_] for p_, id_ in zip(pos, idx)])

        up_x = self.up_decoder(
            self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1),
            x.transpose(0, 1))

        x = up_x.transpose(0, 1)

        if self.upsampling_method == 'bilinear':
            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)

            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)

            if self.num_conv == 2:
                if self.num_upsampe_layer == 2:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
                elif self.num_upsampe_layer == 1:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x, inplace=True)
                    x = self.conv_1(x)
                    x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
            elif self.num_conv == 4:
                if self.num_upsampe_layer == 4:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = self.syncbn_fc_1(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_2(x)
                    x = self.syncbn_fc_2(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_3(x)
                    x = self.syncbn_fc_3(x)
                    x = F.relu(x, inplace=True)
                    x = self.conv_4(x)
                    x = F.interpolate(
                        x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

        return x
