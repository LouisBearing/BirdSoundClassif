# Copyright (c) NBM. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np
from .layers import DepthwiseSepConv2d


class FusionModule(nn.Module):
    
    def __init__(self, n_ends, cn):
        '''
        n_ends: number of inputs
        cn: channels
        '''
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_ends), requires_grad=True)
        self.conv = DepthwiseSepConv2d(cn, cn)
        self.act = nn.ReLU()
    
    def forward(self, inputs):
        '''
        inputs is a list of tensors
        '''
        weights = self.act(self.weights)
        num = 0
        for w, x in zip(weights, inputs):
            num = num + w * x
        den = weights.sum() + 1e-4
        return self.conv(num / den)
    
    
class Rescale(nn.Module):
    
    def __init__(self, in_cn, out_cn):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        if in_cn != out_cn:
            self.pt_wise = nn.Conv2d(in_cn, out_cn, 1)
    def forward(self, x, out_size):
        out = nn.functional.interpolate(x, size=out_size, mode='bilinear', align_corners=True)
        if hasattr(self, 'pt_wise'):
            out = self.pt_wise(out)
        return out
    
    
class BiFPNLayer(nn.Module):
    
    def __init__(self, channels, output_channels=None):
        '''
        2 extremity fmaps, n fmaps in between, the channels are given as input in BOTTOM-UP ORDER
        '''
        super().__init__()
        self.rescalings_td = nn.ModuleDict({
            str(i + 1): Rescale(in_cn, out_cn) for i, (out_cn, in_cn) in enumerate(zip(channels[:-1], channels[1:]))
        })
        self.rescalings_bu = nn.ModuleDict({
            str(i): Rescale(in_cn, out_cn) for i, (in_cn, out_cn) in enumerate(zip(channels[:-1], channels[1:]))
        })
        self.fusions_td = nn.ModuleDict({
            str(i + 1): FusionModule(2, cn) for i, cn in enumerate(channels[1:-1])
        })
        self.fusions_bu = nn.ModuleDict({
            str(i): FusionModule(2 if i in [0, len(channels) - 1] else 3, cn) for i, cn in enumerate(channels)
        })

        if output_channels is not None:
            self.out_pt_wise_convs = nn.ModuleDict({
                str(i): nn.Conv2d(cn, output_channels, 1) for i, cn in enumerate(channels)
            })

    def forward(self, inputs):
        '''
        Params
            inputs: a list (pyramid) of tensors of intermediate backbone activations of decreasing resolution
        Outputs
            bu_outs: a list of tensors of the same sizes as the inputs, with shared semantic meaning between layers
        '''
        sizes = [tuple(e.shape[-2:]) for e in inputs]
        # Top-Down path
        td_indexes = np.arange(len(inputs))[::-1]
        td_out = inputs[-1]
        td_outs = [td_out]
        for i in td_indexes[1:-1]:
            td_out = self.fusions_td[str(i)]([inputs[i], self.rescalings_td[str(i + 1)](td_out, sizes[i])])
            td_outs.insert(0, td_out)
        td_outs.insert(0, self.rescalings_td[str(1)](td_out, sizes[0]))
        
        # Bottom-Up path
        bu_out = self.fusions_bu[str(0)]([inputs[0], td_outs[0]])
        bu_outs = [bu_out]
        for i in np.arange(1, len(inputs) - 1):
            bu_out = self.fusions_bu[str(i)]([inputs[i], td_outs[i], self.rescalings_bu[str(i - 1)](bu_out, sizes[i])])
            bu_outs.append(bu_out)
        bu_outs.append(self.fusions_bu[str(len(inputs) - 1)]([inputs[-1], self.rescalings_bu[str(len(inputs) - 2)](bu_out, sizes[-1])]))

        if hasattr(self, 'out_pt_wise_convs'):
            bu_outs = [self.out_pt_wise_convs[str(i)](bu_out) for i, bu_out in enumerate(bu_outs)]
        
        return bu_outs


class BiFPN(nn.Module):
    
    def __init__(self, n_layers, channels, out_cn):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNLayer(channels, out_cn if i == (n_layers - 1) else None) for i in range(n_layers)])

    def forward(self, x):
        '''
        x is a list of feature maps
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    
'''
Standard FPN / contrary to BiFPN above, here channels of the pyramid are aligned BEFORE going through the FPN
'''
class FPN(nn.Module):

    def __init__(self, channels, p_cn, out_cn):
        super().__init__()
            
        self.pt_wise = nn.ModuleDict({
            str(i): nn.Conv2d(in_channels=cn, out_channels=p_cn, kernel_size=1) for i, cn in enumerate(channels)
        })
        self.out_convs = nn.ModuleDict({
            str(i): nn.Conv2d(in_channels=p_cn, out_channels=out_cn, kernel_size=3, padding=1) for i in range(len(channels))
        })

    def forward(self, x):
        """
        x is a list of intermediate backbone outputs of shapes bs, channels[i], h0/2**i, w0/2**i for index i
        """
        p_outs = [self.pt_wise[str(i)](fm) for (i, fm) in enumerate(x)]
        i = 0
        out = p_outs.pop(-1)
        outs = [self.out_convs[str(i)](out)]
        while len(p_outs) > 0:
            i += 1
            p_out = p_outs.pop(-1)
            upsampled = nn.functional.interpolate(out, size=p_out.shape[-2:], mode='bilinear', align_corners=True)
            out = upsampled + p_out
            outs.insert(0, self.out_convs[str(i)](out))
        return outs
    

def build_fpn(args, channels):
    if args.fpn == 'bifpn':
        model = BiFPN(args.n_bifpn_layers, channels, args.out_fpn_chan)
    elif args.fpn == 'fpn':
        model = FPN(channels, args.fpn_p_chan, args.out_fpn_chan)
    else:
        raise ValueError(f"not supported {args.fpn}")
    return model