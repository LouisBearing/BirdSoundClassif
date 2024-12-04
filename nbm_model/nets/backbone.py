# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding


bcbk_channels = {
    'vgg': {'2': 128, '3': 256, '4': 512, '5': 512},
    'resnet': {'2': 64, '3': 256, '4': 512, '5': 1024, '6': 2048},
    'efficientnet_b0': {'2': 16, '3': 24, '4': 40, '5': 112, '6': 320},
    'efficientnet_b1': {'2': 16, '3': 24, '4': 40, '5': 112, '6': 320},
    'efficientnet_b2': {'2': 16, '3': 24, '4': 48, '5': 120, '6': 352},
    'efficientnet_b3': {'2': 24, '3': 32, '4': 48, '5': 136, '6': 384},
    'efficientnet_b4': {'2': 24, '3': 32, '4': 56, '5': 160, '6': 448},
    'efficientnet_v2_s': {'2': 24, '3': 48, '4': 64, '5': 160, '6': 1280},
    'efficientnet_v2_m': {'2': 24, '3': 48, '4': 80, '5': 176, '6': 1280},
    'efficientnet_v2_l': {'2': 32, '3': 64, '4': 96, '5': 224, '6': 1280}
}

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def is_class(module: nn.Module, query: str):
    return query in module.__class__.__name__.lower()


class BackboneBase(nn.Module): # Arrange for nested tensors

    def __init__(self, backbone: nn.Module, name: str, in_channels: int, train_backbone: bool):
        super().__init__()
        for _, parameter in backbone.named_parameters():
            if not train_backbone: # or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        if is_class(backbone, 'vgg'):
            return_layers = {str(l): str(i + 2) for (i, l) in enumerate([12, 22, 32, 42])}
            self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
            self.num_channels = list(bcbk_channels['vgg'].values())

        elif is_class(backbone, 'resnet'):
            return_layers = {str(l): str(i + 2) for (i, l) in enumerate(['relu', 'layer1', 'layer2', 'layer3', 'layer4'])}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
            self.num_channels = list(bcbk_channels['resnet'].values())

        elif name == 'efficientnet_v2_s':
            return_layers = {str(l): str(i + 2) for (i, l) in enumerate([1, 2, 3, 5, 7])}
            self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
            self.num_channels = list(bcbk_channels[name].values())

        elif 'efficientnet_v2' in name:
            return_layers = {str(l): str(i + 2) for (i, l) in enumerate([1, 2, 3, 5, 8])}
            self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
            self.num_channels = list(bcbk_channels[name].values())

        elif is_class(backbone, 'efficient'):
            return_layers = {str(l): str(i + 2) for (i, l) in enumerate([1, 2, 3, 5, 7])}
            self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
            self.num_channels = list(bcbk_channels[name].values())
        else:
            raise ValueError(f"not supported {backbone.__class__.__name__}")

        if in_channels != 3:
            self.init_conv = nn.Conv2d(in_channels, 3, 1)

        self.strides = [2 ** (i + 1) for i in range(len(self.num_channels))]


    def forward(self, x):
        if hasattr(self, 'init_conv'):
            return self.body(self.init_conv(x))
        return self.body(x)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 in_channels: int,
                 train_backbone: bool,
                 dilation: bool,
                 norm_layer_name: str):
        kwargs = {}
        if (not 'vgg' in name) and (not 'efficientnet_v2' in name):
            if norm_layer_name == 'batchnorm':
                kwargs.update({'norm_layer': nn.BatchNorm2d})
            elif norm_layer_name == 'frozen_batchnorm':
                kwargs.update({'norm_layer': FrozenBatchNorm2d})
        if 'resn' in name:
            kwargs.update({'replace_stride_with_dilation': [False, False, dilation]})
        backbone = getattr(torchvision.models, name)(**kwargs)
        super().__init__(backbone, name, in_channels, train_backbone)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        xs = self[0](x)
        out = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, args.inpt_channels, train_backbone, args.dilation, args.norm_layer_backbone)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    model.strides = backbone.strides
    setattr(args, 'n_layers', len(backbone.num_channels))
    return model
