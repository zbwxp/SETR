import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_up_head import VisionTransformerUpHead, trunc_normal_
from ..builder import HEADS
from ..utils.positional_encoding import PositionEmbeddingSine
from ..utils.transformer import Transformer
from functools import partial
from ..backbones.vit_VT import FilterTokenizer
from mmcv.cnn import build_norm_layer
# from .layers import trunc_normal_

@HEADS.register_module()
class vt_head_direct(VisionTransformerUpHead):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 VT_query=1024,
                 num_ziper_layer=3,
                 **kwargs):
        super(vt_head_direct, self).__init__(**kwargs)

        token_layer = FilterTokenizer(
            in_channels=kwargs['in_channels'],
            token_channels=kwargs['num_classes'],
            tokens=VT_query,
        )
        self.tokenizer = token_layer
        del self.conv_0
        del self.conv_1
        del self.norm


    def forward(self, input):
        x = self._transform_inputs(input)
        x = self.tokenizer(x)

        if x.dim() == 3:
            if x.shape[1] % 32 != 0:
                x = x[:, 1:]

        if self.upsampling_method == 'bilinear':
            if x.dim() == 3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1, 2).reshape(n, c, h, w)

                x = F.interpolate(
                        x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        return x
