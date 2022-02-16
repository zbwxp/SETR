import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_up_head import VisionTransformerUpHead, trunc_normal_
from ..builder import HEADS
from ..utils.positional_encoding import PositionEmbeddingSine
from ..utils.transformer import Transformer
from mmcv.cnn import ConvModule


# from .layers import trunc_normal_

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
class x2down_clean(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 zipee_pos_style=None,
                 ziper_query=256,
                 expand_query=1024,
                 num_ziper_layer=3,
                 num_expand_layer=3,
                 num_heads=12,
                 use_pos=True,
                 **kwargs):
        super(x2down_clean, self).__init__(**kwargs)

        self.use_pos = use_pos
        self.pos_style = zipee_pos_style
        self.num_queries = ziper_query
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

        # self.down = ConvModule(
        #     in_channels=dim,
        #     out_channels=dim,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     norm_cfg=self.norm_cfg
        # )

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.input_proj(x)
        bs = x.size()[0]

        if x.dim() == 3:
            if x.shape[1] % 32 != 0:
                x = x[:, 1:]
            if self.use_pos:
                x += self.zip_pos_embed.pos_table.clone().detach()
            n, hw, c = x.shape
            h = w = int(math.sqrt(hw))
            x = x.transpose(1, 2).reshape(n, c, h, w)
        # down
        down_x = x[:, :, ::2, ::2]
        x = down_x
        x = x.reshape(n, c, hw // 4).transpose(2, 1)
        # up
        down_x = x

        up_x = self.up_decoder(
            self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1),
            down_x.transpose(0, 1))

        x = up_x.transpose(0, 1)

        if self.upsampling_method == 'bilinear':
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
                        x, size=x.shape[-1] * 4, mode='bilinear', align_corners=self.align_corners)
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
                        x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = self.syncbn_fc_1(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_2(x)
                    x = self.syncbn_fc_2(x)
                    x = F.relu(x, inplace=True)
                    x = F.interpolate(
                        x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_3(x)
                    x = self.syncbn_fc_3(x)
                    x = F.relu(x, inplace=True)
                    x = self.conv_4(x)
                    x = F.interpolate(
                        x, size=x.shape[-1] * 2, mode='bilinear', align_corners=self.align_corners)

        return x
