import torch
import torch.nn as nn


class MBConv(nn.Module): # From pytorch repo
    def __init__(self, input_channels, out_channels, kernel, stride, norm_layer, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers = []
        activation_layer = nn.SiLU()

        # expand
        expanded_channels = int(input_channels * expand_ratio)
        if expanded_channels != input_channels:
            layers.extend([
                    nn.Conv2d(input_channels, expanded_channels, kernel_size=1),
                    norm_layer(expanded_channels),
                    activation_layer
            ])

        # depthwise
        layers.extend([
                nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel, stride=stride,
                          padding=int(0.5 * (kernel - 1)), groups=expanded_channels),
                norm_layer(expanded_channels),
                activation_layer
            ])

        # squeeze and excitation
        squeeze_channels = max(1, input_channels // 4)
        layers.append(SqueezeExcitation(expanded_channels, squeeze_channels, activation=nn.SiLU))

        # project
        layers.extend([
                nn.Conv2d(expanded_channels, out_channels, kernel_size=1),
                norm_layer(out_channels)
            ])

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result
    

class FusedMBConv(nn.Module): # From pytorch repo
    def __init__(self, input_channels, out_channels, kernel, stride, norm_layer, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers = []
        activation_layer = nn.SiLU()

        # expand
        expanded_channels = int(input_channels * expand_ratio)
        if expanded_channels != input_channels:
            layers.extend([
                    nn.Conv2d(input_channels, expanded_channels, kernel_size=kernel, stride=stride,
                             padding=int(0.5 * (kernel - 1))),
                    norm_layer(expanded_channels),
                    activation_layer
            ])

            # project
            layers.extend([
                    nn.Conv2d(expanded_channels, out_channels, kernel_size=1),
                    norm_layer(out_channels)
                ])
        else:
            layers.extend([
                    nn.Conv2d(input_channels, out_channels, kernel_size=kernel, stride=stride,
                             padding=int(0.5 * (kernel - 1))),
                    norm_layer(out_channels),
                    activation_layer
            ])            

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result

    
class SqueezeExcitation(torch.nn.Module): # Pytorch implem
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels, squeeze_channels, activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


keys = {
    key: 1 + i for i, key in enumerate(['expand_ratio', 'kernel', 'stride', 'input_channels', 'out_channels', 'num_layers'])
}

inverted_residual_setting = [
    [FusedMBConv, 1, 3, 1, 24, 24, 3],
    [FusedMBConv, 4, 3, 2, 24, 48, 5],
    [FusedMBConv, 4, 3, 2, 48, 80, 5],
    [MBConv, 4, 3, 2, 80, 160, 7],
    [MBConv, 6, 3, 2, 160, 176, 14], # stride = 1 --> stride = 2
    [MBConv, 6, 3, 2, 176, 304, 18],
    [MBConv, 6, 3, 1, 304, 512, 5],
]

class EfficientNet(nn.Module): # From pytorch repo
    # C2d to C1d ? ConvNext ?
    def __init__(self, input_channels, dropout, norm_layer=None):
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0][keys['input_channels']]
        layers.append(nn.Sequential(*[
            nn.Conv2d(input_channels, firstconv_output_channels, kernel_size=3, stride=2, padding=1),
            norm_layer(firstconv_output_channels),
            nn.SiLU()
        ]))

        # building inverted residual blocks
        total_stage_blocks = sum(cnf[keys['num_layers']] for cnf in inverted_residual_setting)
        stage_block_id = 0
        
        for cnf in inverted_residual_setting:
            stage = []
            params = {k:cnf[idx] for k, idx in keys.items() if k != 'num_layers'}
            params.update({'norm_layer': norm_layer})
            
            for _ in range(cnf[keys['num_layers']]):
                stage.append(cnf[0](**params))
                stage_block_id += 1
                params['input_channels'] = params['out_channels']
                params['stride'] = 1

            layers.append(nn.Sequential(*stage))

        self.features = nn.ModuleList(layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        intermediate_outputs = []
        for layer in self.features:
            x = layer(x)
            intermediate_outputs.append(x)

        return intermediate_outputs