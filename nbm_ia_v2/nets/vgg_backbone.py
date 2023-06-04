import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from .layers import *

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        input_channels: int = 1,
        fpn_p_channels: int = 128,
        fpn_o_channels: int = 256,
        fpn: bool = False,
        self_attention: bool = False,
        encode_frequency: bool = False,
        position_encoding: bool = False
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.fpn = fpn
        self.self_attention = self_attention
        self.encode_frequency = encode_frequency
        self.position_encoding = position_encoding
        
        ##
        self.input_channels = input_channels
        self.conv0 = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)

        layers = []
        blocks = []
        for feat in self.features:
            if isinstance(feat, nn.MaxPool2d):
                layers.append(nn.Sequential(*blocks))
                blocks = [feat]
            else:
                blocks.append(feat)
        layers.append(nn.Sequential(*blocks))
        self.layers = layers

        # FPN
        if self.fpn:

            self.up = nn.Upsample(scale_factor=2)

            self.p_convolutions = nn.Sequential(*[
                nn.Conv2d(in_channels=channels + 6 * int(self.encode_frequency), out_channels=fpn_p_channels, kernel_size=1) for channels in [128, 256, 512, 512]
            ])

            self.final_convolutions = nn.Sequential(*[
                nn.Conv2d(in_channels=fpn_p_channels, out_channels=fpn_o_channels, kernel_size=3, padding=1)  for i in range(4)
            ])

        # Self Attention Net
        if self.self_attention:
            self.sa_net = SelfAttention(input_dim=fpn_p_channels, inner_dim=int(fpn_p_channels / 2))
            self.att_weight = nn.Parameter(torch.ones(1).mul_(1), requires_grad=True)
        
        # if init_weights:
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        ### Add frequency encoding
        _, _, height, width = x.shape
        pos_enc_max_channels = self.input_channels # 100
        freq_encoding = torch.matmul(torch.arange(height, dtype=torch.float32).unsqueeze(1), 1 / (1e4 ** (torch.arange(0, pos_enc_max_channels, step=2, dtype=torch.float32) / pos_enc_max_channels)).unsqueeze(0))
        sin = torch.sin(freq_encoding)
        cos = torch.cos(freq_encoding)
        freq_encoding = 0.2 * torch.stack([sin, cos], dim=-1).flatten(start_dim=-2).transpose(0, 1).unsqueeze(-1).repeat(1, 1, width).unsqueeze(0).to(device)
        freq_encoding = freq_encoding[:, :self.input_channels]

        # # x = self.conv0(x + freq_encoding)
        x = x.repeat(1, self.input_channels, 1, 1) + freq_encoding
        x = self.conv0(x)

        out = (x,)

        for block in self.layers:
            x = block(x)
            if self.encode_frequency:
                channels = 3
                dim = x.size(2)
                freq_encoding = torch.matmul(torch.arange(dim, dtype=torch.float32).unsqueeze(1), 1 / (1e3 ** torch.Tensor(np.arange(channels) / channels)).unsqueeze(0))
                sin = torch.sin(freq_encoding)
                cos = torch.cos(freq_encoding)
                freq_encoding = torch.cat([sin, cos], dim=1).transpose(0, 1).view(1, 2 * channels, dim, 1).repeat(x.size(0), 1, 1, x.size(3)).to(device)             
                out += (torch.cat([x, freq_encoding], dim=1),)
            else:
                out += (x,)

        if self.fpn:
            p_outs = [p_conv(out_) for p_conv, out_ in zip(self.p_convolutions, out[-len(self.p_convolutions):])]
            u_out = p_outs[-1]

            if self.self_attention:
                u_out += self.att_weight * self.sa_net(u_out, position_encoding=self.position_encoding).view(u_out.size())
                
            fpn_outs = [self.final_convolutions[-1](u_out)]

            for i in np.arange(len(p_outs) - 2, -1, -1):

                u_out = self.up(u_out) + p_outs[i]

                if (i == len(p_outs) - 2) and (self.self_attention):
                    u_out += self.att_weight * self.sa_net(u_out, position_encoding=self.position_encoding).view(u_out.size())
                    
                fpn_outs.insert(0, self.final_convolutions[i](u_out))
            
            return fpn_outs, out[-1]

        return out[-1]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            self._initialize_module(m)

    def _initialize_module(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512], #'M'
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, pretrained_path: str,  progress: bool, input_channels: int, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), input_channels=input_channels, **kwargs)
    model_dict = model.state_dict()
    if pretrained:
        state_dict = torch.load(pretrained_path)
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def vgg16(pretrained: bool = False, pretrained_path: str = None, progress: bool = True, input_channels: int = 1, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, pretrained_path, progress, input_channels, **kwargs)



def vgg16_bn(pretrained: bool = False, pretrained_path: str = None, progress: bool = True, input_channels: int = 1, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, pretrained_path, progress, input_channels, **kwargs)