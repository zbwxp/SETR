import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

# from .layers import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmcv.cnn import build_norm_layer
import torch
from .vit_up_head import VisionTransformerUpHead, trunc_normal_
from ..utils.positional_encoding import PositionEmbeddingSine
from ..utils.transformer import Transformer

@HEADS.register_module()
class vit_seq_head_pos(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 zipee_pos_style=None,
                 ziper_query=1024,
                 num_ziper_layer=3,
                 **kwargs):
        super(vit_seq_head_pos, self).__init__(**kwargs)

        self.pos_style = zipee_pos_style
        self.num_queries = ziper_query
        hidden_dim = kwargs['embed_dim']

        self.zip_query_embed = nn.Embedding(self.num_queries, hidden_dim)

        N_steps = hidden_dim // 2
        self.zip_pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        sr_ratio = self.num_queries // 1024
        self.zip_transformer = Transformer(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=12,
            dim_feedforward=hidden_dim * 4,
            num_encoder_layers=0,
            num_decoder_layers=num_ziper_layer,
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
            x = self.norm(x)

        src = x
        if self.upsampling_method == 'bilinear':
            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)

            pos = self.zip_pe_layer(x)
            pos = pos.reshape(n, c, hw).transpose(1, 2)
            mask = None
            hs, memory = self.zip_transformer(src, mask, self.zip_query_embed.weight, pos)
            x = hs[0]

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
