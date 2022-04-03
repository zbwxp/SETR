import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_up_head import VisionTransformerUpHead, trunc_normal_
from ..builder import HEADS
from .atm_head import ATMHead

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
class ATM_TPN_roll(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 expand_query=1024,
                 num_expand_layer=1,
                 reverse=False,
                 multi_decoder=False,
                 shrink_query=256,
                 num_ziper_layer=1,
                 num_heads=12,
                 use_norm=False,
                 **kwargs):
        super(ATM_TPN_roll, self).__init__(**kwargs)
        self.reverse = reverse
        self.multi_decoder = multi_decoder
        dim = kwargs['embed_dim']
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
        self.up_decoder = nn.TransformerDecoder(decoder_layer, num_expand_layer)
        if multi_decoder:
            self.up_decoder_1 = nn.TransformerDecoder(decoder_layer, num_expand_layer)
        self.expand_query_embed = nn.Embedding(expand_query, dim)

        self.input_proj_1 = nn.Linear(self.in_channels, dim)
        trunc_normal_(self.input_proj_1.weight, std=.02)

        self.input_proj_2 = nn.Linear(self.in_channels, dim)
        trunc_normal_(self.input_proj_2.weight, std=.02)

        self.use_norm = use_norm
        if self.use_norm:
            self.proj_norm_1 = nn.LayerNorm(dim)
            self.proj_norm_2 = nn.LayerNorm(dim)

        # atm
        self.atm = ATMHead(reverse=True,
                           use_stages=[0],
                           ignore_proj=True,
                           **kwargs,
                           )

    def forward(self, x):
        x1 = x[-1]
        x2 = x[6]
        x1 = self.input_proj_1(x1)
        x2 = self.input_proj_2(x2)
        if self.use_norm:
            x1 = self.proj_norm_1(x1)
            x2 = self.proj_norm_2(x2)
        bs = x1.size()[0]

        if not self.multi_decoder:
            if self.reverse:
                x2 = self.up_decoder(
                    self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1),
                    x2.transpose(0, 1))
                x1 = self.up_decoder(x2, x1.transpose(0, 1))
                x = x1.transpose(0, 1)
            else:
                x1 = self.up_decoder(
                    self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1),
                    x1.transpose(0, 1))
                x2 = self.up_decoder(x1, x2.transpose(0, 1))
                x = x2.transpose(0, 1)
        else:
            if self.reverse:
                x2 = self.up_decoder(
                    self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1),
                    x2.transpose(0, 1))
                x1 = self.up_decoder_1(x2, x1.transpose(0, 1))
                x = x1.transpose(0, 1)
            else:
                x1 = self.up_decoder(
                    self.expand_query_embed.weight.repeat(bs, 1, 1).transpose(0, 1),
                    x1.transpose(0, 1))
                x2 = self.up_decoder_1(x1, x2.transpose(0, 1))
                x = x2.transpose(0, 1)

        atm_out = self.atm([x])

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
        if not self.training:
            return atm_out

        atm_out.update({"aux": x})

        return atm_out
