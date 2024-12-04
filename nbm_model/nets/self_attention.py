# Copyright (c) NBM. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from .position_encoding  import one_dimension_positional_encoding
import copy


class SelfAttention(nn.Module):

    def __init__(self, input_dim, inner_dim, downscale_factor=1, position_encoding=False): # TODO: last param by default
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(input_dim, inner_dim)
        self.key = nn.Linear(input_dim, inner_dim)
        self.value = nn.Linear(input_dim, inner_dim)
        self.final_projection = nn.Linear(inner_dim, input_dim)

        self.downscale_factor = downscale_factor
        self.position_encoding = position_encoding


    def forward(self, inpt):

        bs, input_dim, height_orig, width_orig = inpt.size()
        if self.position_encoding:
            # Shape [1 x input_dim x height_orig x 1] for frequency encoding
            position_encoding = one_dimension_positional_encoding(height_orig, input_dim).transpose(0, 1)[None, ..., None]
            position_encoding = position_encoding.to(inpt.device)
            inpt = inpt + 0.5 * position_encoding

        if self.downscale_factor > 1:
            inpt = nn.functional.interpolate(inpt, size=(height_orig // self.downscale_factor, width_orig // self.downscale_factor), 
                                             mode='bilinear', align_corners=True)

        height, width = inpt.shape[-2:]
        L = height * width

        x = inpt.flatten(start_dim=-2)
        x = x.transpose(1, 2).contiguous().flatten(end_dim=-2)
        
        queries = self.query(x).view(bs, L, -1)
        keys = self.key(x).view(bs, L, -1)
        values = self.value(x).view(bs, L, -1)
        
        factors = torch.softmax(torch.matmul(queries, keys.transpose(1, 2)) / np.round(np.sqrt(queries.size(-1)), 2), dim=-1)
        context_vect = torch.matmul(factors, values)
        context_vect = self.final_projection(context_vect.flatten(end_dim=-2)).view(bs, L, input_dim).transpose(1, 2)\
                .contiguous().view(bs, input_dim, height, width)

        if self.downscale_factor > 1:
            context_vect = nn.functional.interpolate(inpt, size=(height_orig, width_orig), 
                                                     mode='bilinear', align_corners=True)

        return context_vect


class SAPyramid(nn.Module):

    def __init__(self, channels, top_n):
        super().__init__()
        if top_n == len(channels): # TODO Change this by default
            self.attention_modules = nn.ModuleDict({
                str(i): SelfAttention(cn, cn, max(1, 2 ** (3 - i)), True) for (i, cn) in enumerate(channels)
            })
        else:
            self.attention_modules = nn.ModuleDict({
                str(i): SelfAttention(cn, cn // 2) if (i >= (len(channels) - top_n)) else nn.Identity() for (i, cn) in enumerate(channels)
            })

    def forward(self, x):
        '''
        x is a bottom-up ordered list of backbone outputs
        '''
        return [fm + self.attention_modules[str(i)](fm) for (i, fm) in enumerate(x)]


def build_sa_layers(args, channels):
    if type(channels) == tuple:
        return nn.ModuleList([SAPyramid(cns, args.pyramid_top_n_attn) for cns in channels])
    return SAPyramid(channels, args.pyramid_top_n_attn)

###
## Transformer layers adapted from Detr
###


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor + pos

    def forward_post(self, src, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos):
        return self.forward_post(src, pos)