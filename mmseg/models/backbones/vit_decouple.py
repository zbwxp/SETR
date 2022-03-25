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

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



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
class vit_decouple(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 shrink_index=4,
                 expand_index=100,
                 num_queries=256,
                 use_norm=True,
                 **kwargs):
        super(vit_decouple, self).__init__(**kwargs)
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
        decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 4)
        self.decoder = TPN_Decoder(decoder_layer, num_expand_layer)
        self.q = nn.Embedding(num_queries, dim)
        self.register_buffer("init_once", torch.tensor(0))
        self.register_buffer("_iter", torch.tensor(0))
        self.zip_pos_embed = PositionalEncoding(kwargs['embed_dim'], num_queries)
        self.attn_proj = nn.Linear(dim, dim)
        trunc_normal_(self.attn_proj.weight, std=.02)

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
            x = blk(x)
            if i == self.shrink_index:
                # if self.use_norm:
                #     x = x[:, 1:]
                #     x = self.shrink_norm(x)
                x_ = x.clone().detach()
                # supervise right before
                outs.append(x)
                bs, num_token, ch = x.size()
                if self.init_once == 0:
                    self.init_once += 1
                    print("self.q init once!!!!!!!")
                    idx = torch.randint(1, num_token, (self.q.num_embeddings,))
                    init_param = x[:, idx]
                    self.q.weight = nn.Parameter(init_param[0])
                x, attn = self.decoder(self.q.weight.repeat(bs, 1, 1).transpose(0, 1), x.transpose(0, 1))
                # attn = attn.sigmoid()
                x = x.transpose(0, 1)
                cos = nn.CosineSimilarity(dim=2)
                sim_attn = [cos(attn, attn[:, i][:, None]) for i in range(self.q.num_embeddings)]
                loss_attn = torch.stack(sim_attn, dim=1)
                sim_x = [cos(x, x[:, i][:, None]) for i in range(self.q.num_embeddings)]
                loss_x = torch.stack(sim_x, dim=1)
                loss_sim = (loss_attn + loss_x).mean()
                # if self._iter % 200 == 0:
                print(loss_sim)
            if i in self.out_indices:
                if attn is not None:
                    val, idx = attn.max(-2)
                    out = torch.stack([x_i[idx_i] for x_i, idx_i in zip(x, idx)]) * val[:, :, None]
                    if i == self.shrink_index:
                        loss = nn.MSELoss()
                        loss_mse = loss(self.attn_proj(out), x_)
                    outs.append(out)
                else:
                    outs.append(x)
        outs.append({"loss_similarity": loss_sim * 10.0})

        return tuple(outs)
