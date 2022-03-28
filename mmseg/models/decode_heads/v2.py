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
class v2(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 expand_query=1024,
                 num_ziper_layer=3,
                 use_rand_idx=True,
                 num_expand_layer=3,
                 num_heads=12,
                 use_norm=False,
                 shrink="2x2",
                 **kwargs):
        super(v2, self).__init__(**kwargs)

        self.use_idx = use_rand_idx
        dim = kwargs['in_channels'] // 2
        self.shrink = shrink

        self.input_proj = nn.Linear(self.in_channels, dim)
        trunc_normal_(self.input_proj.weight, std=.02)
        self.base_proj = nn.Linear(self.in_channels, dim)
        trunc_normal_(self.base_proj.weight, std=.02)

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.use_norm = use_norm
        if self.use_norm:
            self.proj_norm = nn.LayerNorm(dim)

        if self.upsampling_method == '2x2_distributed':
            del self.conv_0
            self.conv_0 = nn.Conv2d(dim // 4, 256, 1, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        base = x[2][:, 1:].detach()
        base = self.base_proj(base)
        x = self._transform_inputs(x)
        x = self.input_proj(x)
        if self.use_norm:
            x = self.proj_norm(x)
        B, Nq, C = x.size()
        Nk = base.size()[1]

        q = self.q(x).reshape(B, Nq, self.num_heads,
                               C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(base).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn.sum(dim=1).transpose(-1, -2)

        if self.upsampling_method == 'bilinear':
            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)

            x = self.conv_0(x)
            x = self.syncbn_fc_0(x)
            x = F.relu(x, inplace=True)
            x = self.conv_1(x)
            x = F.interpolate(
                x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)

        elif self.upsampling_method == '2x2_distributed':
            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)
                x = x.reshape(n, 2, 2, c // 4, h, w).permute(0,3,1,4,2,5).reshape(n, c // 4, 2*h, 2*w)

            x = self.conv_0(x)
            x = self.syncbn_fc_0(x)
            x = F.relu(x, inplace=True)
            x = self.conv_1(x)
            x = F.interpolate(
                x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)

        return x
