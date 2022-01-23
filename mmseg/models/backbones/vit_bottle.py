import torch
import torch.nn as nn
import math

from .vit import VisionTransformer
from ..builder import BACKBONES
from ..utils.positional_encoding import PositionEmbeddingSine
from ..utils.transformer import Transformer
# from .layers import DropPath, to_2tuple, trunc_normal_

@BACKBONES.register_module()
class Vit_Bottleneck(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 zipee_pos_style=None,
                 ziper_index=7,
                 ziper_query=256,
                 num_ziper_layer=3,
                 **kwargs):
        super(Vit_Bottleneck, self).__init__(**kwargs)

        self.pos_style = zipee_pos_style
        self.ziper_index = ziper_index
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
            if i == self.ziper_index:
                if self.pos_style is None:
                    pos = torch.zeros_like(self.pos_embed)
                elif self.pos_style == "orig":
                    pos = self.pos_embed
                elif self.pos_style == "sine":
                    x = x[:, 1:]
                    n, hw, c = x.shape
                    h = w = int(math.sqrt(hw))
                    pos = self.zip_pe_layer(x.transpose(1, 2).reshape(n, c, h, w))
                    pos = pos.flatten(2).transpose(1, 2)
                src = x
                mask = None
                hs, attn = self.zip_transformer(src, mask, self.zip_query_embed.weight, pos)
                x = hs[0]
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        outs.append(attn)
        return tuple(outs)
