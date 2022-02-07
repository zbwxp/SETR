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
class vit_seq_head_pos_sincos_convdownup_v2(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 zipee_pos_style=None,
                 ziper_query=256,
                 expand_query=1024,
                 num_ziper_layer=3,
                 num_expand_layer=3,
                 **kwargs):
        super(vit_seq_head_pos_sincos_convdownup_v2, self).__init__(**kwargs)

        self.pos_style = zipee_pos_style
        self.num_queries = ziper_query
        hidden_dim = kwargs['embed_dim']

        # self.zip_query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.expand_query_embed = nn.Embedding(expand_query, hidden_dim)
        N_steps = hidden_dim // 2
        # self.zip_pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # self.zip_pos_embed = PositionalEncoding(kwargs['embed_dim'], 1024)
        self.expand_pos_embed = PositionalEncoding(kwargs['embed_dim'], ziper_query)
        sr_ratio = self.num_queries // 1024
        # self.zip_transformer = Transformer(
        #     d_model=hidden_dim,
        #     dropout=0.1,
        #     nhead=12,
        #     dim_feedforward=hidden_dim * 4,
        #     num_encoder_layers=0,
        #     num_decoder_layers=num_ziper_layer,
        #     normalize_before=False,
        #     return_intermediate_dec=False,
        #     sr_ratio=sr_ratio,
        # )

        self.down = ConvModule(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg
        )

        self.expand_transformer = Transformer(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=12,
            dim_feedforward=hidden_dim * 4,
            num_encoder_layers=0,
            num_decoder_layers=num_expand_layer,
            normalize_before=False,
            return_intermediate_dec=False,
            sr_ratio=sr_ratio,
        )

        if self.in_channels != hidden_dim:
            self.input_proj = nn.Linear(self.in_channels, hidden_dim)
            trunc_normal_(self.input_proj.weight, std=.02)
        else:
            self.input_proj = nn.Sequential()

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.input_proj(x)
        if x.dim() == 3:
            if x.shape[1] % 32 != 0:
                x = x[:, 1:]
            n, hw, c = x.shape
            h = w = int(math.sqrt(hw))
            x = x.transpose(1, 2).reshape(n, c, h, w)
        # down
        x = self.down(x)
        x = x.reshape(n, c, hw // 4).transpose(2, 1)
        # up
        pos = self.expand_pos_embed.pos_table.clone().detach()
        src = x
        mask = None
        hs, memory = self.expand_transformer(src, mask, self.expand_query_embed.weight, pos)
        x = hs[0]

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
