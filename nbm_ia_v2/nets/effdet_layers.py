import torch
import torch.nn as nn
import numpy as np
import os
# from .effnet import *
from torchvision.models import efficientnet
from .nets_utils import *


class InvResX1D(nn.Module):
    """
    Inverted Residual Block 1D - ConvNeXt style. Applies LayerNorm on the channel dimension.
    """

    def __init__(self, indim, outdim, kernel=3, stride=1, expansion_fact=4, residual=True, act_out=True, bias_out=True):
        super(InvResX1D, self).__init__()
        self.depth_wise = nn.Conv1d(indim, indim, kernel, stride=stride, padding=int(0.5 * (kernel - 1)),
                                    groups=indim)
        self.norm = nn.LayerNorm(indim)
        self.pt_wise_in = nn.Conv1d(indim, expansion_fact * indim, 1)
        self.act = nn.GELU()
        self.pt_wise_out = nn.Conv1d(expansion_fact * indim, outdim, 1, bias=bias_out)
        
        self.residual = residual
        self.downsample = None
        if residual and ((stride != 1) or (indim != outdim)):
            self.downsample = nn.Conv1d(indim, outdim, 1, stride, groups=groups)
            self.out_norm = nn.LayerNorm(outdim)

        self.act_out = act_out
        
    def forward(self, x):
        # Expected shape: B x C x D, resp batch size, channels, dimension
        identity = x
        out = self.depth_wise(x)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.pt_wise_in(out)
        out = self.act(out)
        out = self.pt_wise_out(out)
        
        if not self.residual:
            if self.act_out:
                out = self.act(out)
            return out
        
        if self.downsample is not None:
            identity = self.out_norm(self.downsample(x).transpose(1, 2)).transpose(1, 2)
            
        out += identity
        if self.act_out:
            out = self.act(out)
            
        return out


class InvResX2D(nn.Module):
    """
    Inverted Residual Block 2D - ConvNeXt style
    """

    def __init__(self, indim, outdim, kernel=3, stride=1, expansion_fact=4, residual=True, act_out=True, bias_out=True):
        super(InvResX2D, self).__init__()
        if type(kernel) == tuple:
            padding = (int(0.5 * (kernel[0] - 1)), int(0.5 * (kernel[1] - 1)))
        else:
            padding = int(0.5 * (kernel - 1))
        self.depth_wise = nn.Conv2d(indim, indim, kernel, stride=stride, padding=padding,
                                    groups=indim)
        # self.norm = nn.BatchNorm2d(indim)
        self.norm = nn.LayerNorm(indim)
        self.pt_wise_in = nn.Conv2d(indim, expansion_fact * indim, 1)
        self.act = nn.GELU()
        self.pt_wise_out = nn.Conv2d(expansion_fact * indim, outdim, 1, bias=bias_out)

        self.residual = residual
        self.downsample = None
        if residual and ((stride != 1) or (indim != outdim)):
            self.downsample = nn.Conv2d(indim, outdim, 1, stride)
                # nn.BatchNorm2d(outdim)
            self.out_norm = nn.LayerNorm(outdim)
        
        self.act_out = act_out
        
    def forward(self, x):
        # Expected shape: B x C x H x W
        bs, chan, h, w = x.shape
        identity = x
        out = self.depth_wise(x)
        # out = self.norm(out)
        out = self.norm(out.flatten(start_dim=-2).transpose(1, 2)).transpose(1, 2).view(bs, chan, h, w)
        out = self.pt_wise_in(out)
        out = self.act(out)
        out = self.pt_wise_out(out)

        if not self.residual:
            if self.act_out:
                out = self.act(out)
            return out
        
        if self.downsample is not None:
            # identity = self.downsample(x)
            identity = self.out_norm(self.downsample(x).flatten(start_dim=-2).transpose(1, 2)).transpose(1, 2).view(bs, -1, h, w)
            
        out += identity
        if self.act_out:
            out = self.act(out)
            
        return out


class SmallViT(nn.Module):
    
    def  __init__(self, input_dim, patch_h, patch_w, dropout=0.1, nhead=8, dim_feedforward=1024, num_layers=1):
        super().__init__()
        
        self.patch_h, self.patch_w = patch_h, patch_w
        
        hidden_dim = input_dim * patch_h * patch_w
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        
        transf_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout)
        self.encoder = nn.TransformerEncoder(transf_layer, num_layers=num_layers)
        
    def forward(self, x):
        bs, c, h, w = x.shape
        n_h, n_w = int(h / self.patch_h), int(w / self.patch_w)
        
        x_seq = self.proj(x)
        out = self.encoder(x_seq.flatten(start_dim=2).permute(2, 0, 1)).permute(1, 2, 0)
        out = out.view(bs, c, self.patch_h, self.patch_w, n_h, n_w).permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, c, h, w)
        
        return out


class IntermEffNet(efficientnet.EfficientNet):
    
    def forward(self, x):
        
        intermediate_outputs = []
        for layer in self.features:
            x = layer(x)
            intermediate_outputs.append(x)

        return intermediate_outputs


def effnet_with_interm(backbone):
    
    effnet_config = effnet_configs[backbone]
    inverted_residual_setting, last_channel = efficientnet._efficientnet_conf(effnet_config['name'], width_mult=effnet_config['width_mult'], 
        depth_mult=effnet_config['depth_mult'])
    
    return IntermEffNet(inverted_residual_setting=inverted_residual_setting, dropout=effnet_config['dropout'], last_channel=last_channel)


class FusionModule(nn.Module):
    
    def __init__(self, n_ends, cn, expansion_fact=4):
        '''
        n_ends: number of inputs
        cn: channels
        '''
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_ends), requires_grad=True)
        self.conv = InvResX2D(cn, cn, residual=True, expansion_fact=expansion_fact)
        # self.conv = nn.Conv2d(cn, cn, 3, padding=1)
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
    
    def __init__(self, channels, output_channels=None, expansion_fact=4):
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
            str(i + 1): FusionModule(2, cn, expansion_fact) for i, cn in enumerate(channels[1:-1])
        })
        self.fusions_bu = nn.ModuleDict({
            str(i): FusionModule(2 if i in [0, len(channels) - 1] else 3, cn, expansion_fact) for i, cn in enumerate(channels)
        })

        if output_channels is not None:
            self.out_pt_wise_convs = nn.ModuleDict({
                str(i): nn.Conv2d(cn, output_channels[i], 1) for i, cn in enumerate(channels)
            })
        
        # self.out_pt_wise_convs = nn.ModuleDict({
        #         str(i): nn.Conv2d(cn, output_channels[i], 1) for i, cn in enumerate(channels)
        #     })

    def forward(self, inputs, sizes):
        # td path
        td_indexes = np.arange(len(inputs))[::-1]
        td_out = inputs[-1]
        td_outs = [td_out]
        for i in td_indexes[1:-1]:
            td_out = self.fusions_td[str(i)]([inputs[i], self.rescalings_td[str(i + 1)](td_out, sizes[i])])
            td_outs.insert(0, td_out)
        td_outs.insert(0, self.rescalings_td[str(1)](td_out, sizes[0]))
        
        # bu path
        bu_out = self.fusions_bu[str(0)]([inputs[0], td_outs[0]])
        bu_outs = [bu_out]
        for i in np.arange(1, len(inputs) - 1):
            bu_out = self.fusions_bu[str(i)]([inputs[i], td_outs[i], self.rescalings_bu[str(i - 1)](bu_out, sizes[i])])
            bu_outs.append(bu_out)
        bu_outs.append(self.fusions_bu[str(len(inputs) - 1)]([inputs[-1], self.rescalings_bu[str(len(inputs) - 2)](bu_out, sizes[-1])]))

        if hasattr(self, 'out_pt_wise_convs'):
            bu_outs = [self.out_pt_wise_convs[str(i)](bu_out) for i, bu_out in enumerate(bu_outs)]
        # bu_outs = [self.out_pt_wise_convs[str(i)](bu_out) for i, bu_out in enumerate(inputs)]
        
        return bu_outs


class BiFPN(nn.Module):
    
    def __init__(self, n_layers, channels, out_channels, expansion_fact):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNLayer(channels, out_channels if i == (n_layers - 1) else None, expansion_fact) for i in range(n_layers)])
        # self.layers = BiFPNLayer(channels, out_channels, expansion_fact)

    def forward(self, x, sizes):
        '''
        x is a list of feature maps
        '''
        for layer in self.layers:
            x = layer(x, sizes)
        # x = self.layers(x, sizes)
        return x


class ClassBoxBranches(nn.Module):
    
    def __init__(self, channels, n_anchors, n_layers_branches, n_classes, c1d_branches, bias_p, bckb):
        super().__init__()

        self.n_classes = n_classes
        self.c1d_branches = c1d_branches
        self.bckb = bckb
        bias_out_bool = True if bias_p == 0 else False
        
        if c1d_branches:
            self.class_branch = nn.ModuleList([
                nn.Sequential(*[
                    InvResX1D(chan_lvl, chan_lvl, expansion_fact=2) if i < (n_layers_branches - 1) \
                    else InvResX1D(chan_lvl, n_anchors * n_classes * SIZES[self.bckb][lvl][0], expansion_fact=2, residual=False, act_out=False, bias_out=bias_out_bool) \
                    for i in range(n_layers_branches)
                ]) for (lvl, chan_lvl) in enumerate(channels)
            ])

            self.box_branch = nn.ModuleList([
                nn.Sequential(*[
                    InvResX1D(chan_lvl, chan_lvl, expansion_fact=2) if i < (n_layers_branches - 1) \
                    else InvResX1D(chan_lvl, n_anchors * 4 * SIZES[self.bckb][lvl][0], expansion_fact=2, residual=False, act_out=False) \
                    for i in range(n_layers_branches)
                ]) for (lvl, chan_lvl) in enumerate(channels)
            ])

        else:
            self.class_branch = nn.ModuleList([
                nn.Sequential(*[
                    InvResX2D(chan_lvl, chan_lvl) if i < (n_layers_branches - 1) \
                    else InvResX2D(chan_lvl, n_anchors * n_classes, residual=False, act_out=False, bias_out=bias_out_bool) \
                    for i in range(n_layers_branches)
                ]) for (lvl, chan_lvl) in enumerate(channels)
            ])
            self.box_branch = nn.ModuleList([
                nn.Sequential(*[
                    InvResX2D(chan_lvl, chan_lvl) if i < (n_layers_branches - 1) \
                    else InvResX2D(chan_lvl, n_anchors * 4, residual=False, act_out=False, bias_out=bias_out_bool) \
                    for i in range(n_layers_branches)
                ]) for (lvl, chan_lvl) in enumerate(channels)
            ])

            # self.class_branch = nn.Sequential(*[InvResX2D(channels, channels) if i < (n_layers_branches - 1) \
            #     else InvResX2D(channels, n_anchors * n_classes, expansion_fact=2, residual=False, act_out=False, bias_out=bias_out_bool) for i in range(n_layers_branches)])
            # self.box_branch = nn.Sequential(*[InvResX2D(channels, channels) if i < (n_layers_branches - 1) \
            #     else InvResX2D(channels, n_anchors * 4, expansion_fact=2, residual=False, act_out=False) for i in range(n_layers_branches)])

        
        self.class_branch_act = nn.Softmax(dim=2)

        if bias_p > 0:
            self._initialize_out_bias(c1d_branches, bias_p)


    def forward(self, out_fpn):
        if self.c1d_branches:
            bs = len(out_fpn[0])
            boxes = [
                self.box_branch[i](o.flatten(start_dim=1, end_dim=2)).view(bs, -1, SIZES[self.bckb][i][0], SIZES[self.bckb][i][1])
                for (i, o) in enumerate(out_fpn)
            ]
            if hasattr(self, 'out_bias'):
                classes = [
                    (self.class_branch[i](o.flatten(start_dim=1, end_dim=2)) + self.out_bias[i][None, :, None]).view(bs, -1, SIZES[self.bckb][i][0], SIZES[self.bckb][i][1])
                    for (i, o) in enumerate(out_fpn)
                ]
            else:
                classes = [
                    (self.class_branch[i](o.flatten(start_dim=1, end_dim=2))).view(bs, -1, SIZES[self.bckb][i][0], SIZES[self.bckb][i][1])
                    for (i, o) in enumerate(out_fpn)
                ]
        else:
            boxes = [self.box_branch[i](o) for (i, o) in enumerate(out_fpn)]
            if hasattr(self, 'out_bias'):
                classes = [self.class_branch[i](o) + self.out_bias[None, :, None, None] for (i, o) in enumerate(out_fpn)]
            else:
                classes = [self.class_branch[i](o) for (i, o) in enumerate(out_fpn)]
            # boxes = [self.box_branch(o) for o in out_fpn]
            # if hasattr(self, 'out_bias'):
            #     classes = [self.class_branch(o) + self.out_bias[None, :, None, None] for o in out_fpn]
            # else:
            #     classes = [self.class_branch(o) for o in out_fpn]

        classes = [
            self.class_branch_act(
                classes_lvl.view(-1, N_ANCHORS, self.n_classes, classes_lvl.shape[-2], classes_lvl.shape[-1])
            ).flatten(start_dim=1, end_dim=2) for classes_lvl in classes
        ]

        return classes, boxes


    def _initialize_out_bias(self, c1d_branches, p=0.01):
        logit_bias = -np.log((1 - p) / p)
        bias = torch.ones(self.n_classes) * logit_bias
        bias[0] = -np.log(p / (1 - p))
        bias = bias.repeat(N_ANCHORS)
        if c1d_branches:
            self.out_bias = nn.ParameterList([
                nn.Parameter(bias.repeat_interleave(SIZES[self.bckb][i][0]), requires_grad=True) for i in range(len(self.class_branch))
            ])
        else:
            self.out_bias = nn.Parameter(bias, requires_grad=True)


class EffDet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.init_lay = nn.Conv2d(1, 3, 1)
        self.backbone = effnet_with_interm(config.bckb)
        if config.pretrained_bckb:
            state_dict = torch.load(os.path.join(config.pretrain_path, pretr_mods[config.bckb]))
            self.backbone.load_state_dict(state_dict)
            self.backbone.train()
        # if self.config.bckb == 'effnet':
        #     self.backbone = EfficientNet(1, config.dropout)
        # elif self.config.bckb == 'vgg':
        #     self.backbone = vgg16_bn(pretrained=config.pretrained_bckb, pretrained_path=config.pretrain_path)
        # if config.c1d_branches:
        #     out_channels_fpn = [int(chan / config.c1d_div_fact) for chan in CHANNELS[self.config.bckb]]
        #     int_channels_branches = [chan * size[0] for chan, size in zip(out_channels_fpn, SIZES[self.config.bckb])]
        # else:
        #     # out_channels_fpn = [config.out_channels] * len(CHANNELS[self.config.bckb])
        out_channels_fpn = CHANNELS[self.config.bckb]
        int_channels_branches = out_channels_fpn
        #     # int_channels_branches = config.out_channels
        self.fpn = BiFPN(config.n_layers_bifpn, CHANNELS[self.config.bckb], out_channels_fpn, config.expansion_fact_fpn)
        self.branches = ClassBoxBranches(int_channels_branches, N_ANCHORS, config.n_layers_branches, 1 + config.n_classes, config.c1d_branches, config.bias_p,
            config.bckb)
        if config.pre_fpn_attn:
            patch_size = (2, 8)
            self.attn_modules = nn.ModuleDict({
                str(i): SmallViT(CHANNELS[self.config.bckb][i], max(1, int(patch_size[0] / (2 ** i))),
                    max(1, int(patch_size[1] / (2 ** i))), dropout=config.dropout) for i in range(len(CHANNELS[self.config.bckb]))
            })

        self.anchor_tgt_layer = AnchorTargetLayer(config)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            self._initialize_module(m)
        # if self.config.bias_p > 0:
        #     self._initialize_class_branch_bias(self.config.bias_p)

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

    # def _initialize_class_branch_bias(self, p=0.01):
    #     logit_bias = -np.log((1 - p) / p)
    #     bias = torch.ones(self.config.n_classes + 1).to(self.config.device) * logit_bias
    #     bias[0] = -np.log(p / (1 - p))
    #     bias = bias.repeat(N_ANCHORS)
    #     if self.config.c1d_branches:
    #         for i in range(len(self.branches.class_branch)):
    #             self.branches.class_branch[i][-1].pt_wise_out.bias.data = bias.repeat_interleave(SIZES[self.config.bckb][i][0])
    #     else:
    #         self.branches.class_branch[-1].pt_wise_out.bias.data = bias

    def forward(self, inputs):
        '''
        inputs shape: bs, h, w
        '''
        out = self.backbone(self.init_lay(inputs[:, None]))
        out = [out[i] for i in BCKB_LAYERS[self.config.bckb]]
        if self.config.pre_fpn_attn:
            out = [inpt_lvl + self.attn_modules[str(i)](inpt_lvl) for (i, inpt_lvl) in enumerate(out)]
        out = self.fpn(out, SIZES[self.config.bckb])
        classes, boxes = self.branches(out)
        
        return classes, boxes

    def step(self, batch, neg_step=False, alpha=None, gamma=None):

        img, neg_img, bb_coord, bird_ids, lengths = batch
        img, neg_img, bb_coord, bird_ids = img.to(self.config.device), neg_img.to(self.config.device), bb_coord.to(self.config.device), bird_ids.to(self.config.device)
        b_size = len(lengths)

        if not neg_step:
            ### Positive batch step

            # Forward & reshape
            classes, boxes = self(img)
            classes = [classes_lvl.view(b_size, N_ANCHORS, 1 + self.config.n_classes, classes_lvl.shape[-2], classes_lvl.shape[-1])\
                .permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=3) for classes_lvl in classes]
            boxes = [bbox_lvl.view(b_size, N_ANCHORS, 4, bbox_lvl.shape[-2], bbox_lvl.shape[-1])\
                .permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=3) for bbox_lvl in boxes]
            # Target layer
            labels, reg_targets = self.anchor_tgt_layer(bb_coord, bird_ids, lengths, loss_type='focal') # or frcnn
            gt_class_logits = [torch.gather(classes_lvl, 2, label_lvl[..., None].clamp(0)).squeeze(-1) for classes_lvl, label_lvl in zip(classes, labels)]
            # Flatten cls & labels
            gt_class_logits = torch.cat(gt_class_logits, dim=1)
            labels = torch.cat(labels, dim=1)
            # Flatten boxes & regression targets
            boxes = torch.cat(boxes, dim=1)
            reg_targets = torch.cat(reg_targets, dim=1)
            # Modulate losses wrt pos / neg / ignore assignments
            ignore_idx = (labels == -1).to(torch.float)
            pos_idx = labels > 0
            # ignore_idx = extract_samples_from_labels(labels, self.config.n_samples_ce)
            # Losses
            # cls_loss = cross_entropy_loss(gt_class_logits, ignore_idx, self.config.n_samples_ce)
            cls_loss = focal_loss(gt_class_logits, ignore_idx, pos_idx, alpha if alpha is not None else self.config.alpha, 
                gamma if gamma is not None else self.config.gamma)
            reg_loss = smooth_l1_loss(boxes, reg_targets, pos_idx)

            neg_cls_loss = torch.tensor(0.0).to(self.config.device)

        else:
            ### Negative batch step
            neg_classes, _ = self(neg_img)
            neg_logits = [classes_lvl.view(b_size, N_ANCHORS, 1 + self.config.n_classes, classes_lvl.shape[-2], classes_lvl.shape[-1])[:, :, 0] for classes_lvl in neg_classes]
            neg_logits = torch.cat([logits_lvl.flatten(start_dim=1) for logits_lvl in neg_logits], dim=1)
            neg_cls_loss = focal_loss_neg(neg_logits, self.config.gamma, alpha if alpha is not None else self.config.alpha)

            cls_loss = torch.tensor(0.0).to(self.config.device)
            reg_loss = torch.tensor(0.0).to(self.config.device)          

        return cls_loss.mean(), reg_loss.mean(), neg_cls_loss.mean()

    def eval_test(self, inputs, min_score=0.5):
        with torch.no_grad():
            classes, boxes = self(inputs)

        return self.infer_cls_boxes(classes, boxes, min_score)
    
    def infer_cls_boxes(self, classes, boxes, min_score):
        """
        Takes as input the class and box coord predictions from the model, filters w.r.t. min scores, and applies non-maximum suppressions
        """

        batch_size = len(classes[0])

        # suppress class zero (background)
        max_classes = [classes_lvl.view(batch_size, N_ANCHORS, 1 + self.config.n_classes, classes_lvl.shape[-2],
             classes_lvl.shape[-1])[:, :, 1:].max(dim=2) for classes_lvl in classes]
        scores, predicted_classes = zip(*max_classes)

        # Flatten
        scores = torch.cat([scores_lvl.flatten(start_dim=1) for scores_lvl in scores], dim=-1)
        predicted_classes = 1 + torch.cat([cls_lvl.flatten(start_dim=1) for cls_lvl in predicted_classes], dim=-1)

        # Boxes
        boxes = [bbox_lvl.view(batch_size, N_ANCHORS, 4, bbox_lvl.shape[-2],
             bbox_lvl.shape[-1]).permute(0, 1, 3, 4, 2).flatten(start_dim=1, end_dim=3) for bbox_lvl in boxes]

        # Reg to box coord
        boxes = [bbox_reg_to_coord(boxes[i], self.anchor_tgt_layer.anchors[i]) for i in range(len(boxes))]
        boxes = torch.cat(boxes, dim=1)

        # Clip bbox proposals to image
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(min=0, max=IMG_SIZE[1] - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(min=0, max=IMG_SIZE[0] - 1)

        # Sort by decreasing confidence
        sorted_idx = scores.argsort(descending=True)
        sorted_scores = torch.gather(scores, 1, sorted_idx)
        sorted_boxes = torch.gather(boxes, 1, sorted_idx[..., None].repeat(1, 1, 4))
        sorted_classes = torch.gather(predicted_classes, 1, sorted_idx)

        output = []
        # Iterate batch and append final results
        for b_idx in range(batch_size):

            b_output = {}

            # First NMS, all classes + suppress class 0
            non_zeros_where = (sorted_scores[b_idx] >= min_score) # & (sorted_classes[b_idx] > 0)
            if not non_zeros_where.any():
                output.append(
                    {str(class_idx): dict(bbox_coord=torch.Tensor(), scores=torch.Tensor()) for class_idx in range(1, self.config.n_classes + 1)}
                )
                continue

            b_sorted_boxes = sorted_boxes[b_idx, non_zeros_where][:self.config.pre_nms_topN].unsqueeze(0)
            b_sorted_scores = sorted_scores[b_idx, non_zeros_where][:self.config.pre_nms_topN].unsqueeze(0)
            b_sorted_classes = sorted_classes[b_idx, non_zeros_where][:self.config.pre_nms_topN]

            # NMS
            b_sorted_boxes, b_sorted_scores, nms_idx = nms(b_sorted_boxes, b_sorted_scores, post_nms_topN=len(b_sorted_classes), 
                nms_thresh=self.config.inter_nms_thresh, return_idx=True)

            b_sorted_boxes = b_sorted_boxes[0]
            b_sorted_scores = b_sorted_scores[0]
            b_sorted_classes = b_sorted_classes[nms_idx]

            # Apply NMS separately for each class
            for class_idx in range(1, self.config.n_classes + 1): # class "other" ??
                class_where = b_sorted_classes == class_idx

                if not class_where.any():
                    b_output[str(class_idx)] = dict(
                        bbox_coord=torch.Tensor(), 
                        scores=torch.Tensor())
                    continue

                nms_bbox_inpt = b_sorted_boxes[class_where].unsqueeze(0)
                nms_scores_inpt = b_sorted_scores[class_where].unsqueeze(0)

                class_boxes, class_scores = nms(nms_bbox_inpt, nms_scores_inpt, post_nms_topN=500, nms_thresh=self.config.intra_nms_thresh)
                class_boxes = class_boxes.view(-1, 4)
                # class_scores = class_scores.flatten()

                b_output[str(class_idx)] = dict(
                    bbox_coord=class_boxes,
                    scores=class_scores
                )

            output.append(b_output)

        return output


class AnchorTargetLayer(nn.Module):
        
    def __init__(self, config):
        super().__init__()

        self.config = config
        n_scales = len(CHANNELS[config.bckb])
        # n_scales = config.max_level - config.min_level + 1
        
        # Generate anchors -> list of anchor box for all levels
        anchors = generate_anchors(config.base_size, config.ratios, n_scales)
        
        # Move anchors over spatial coordinates
        anchors_shifts = get_anchor_shifts(config.bckb)
        all_anchors = [(anch + shifts).reshape(-1, 4) for anch, shifts in zip(anchors, anchors_shifts)]
        self.anchors = [torch.Tensor(all_anch).to(config.device) for all_anch in all_anchors]
    
    
    def forward(self, bb_coord, bird_ids, lengths, loss_type='focal'):
        """
        Generates regression objectives and classification labels (-1 for ignored samples)
        related to each anchor, given the ground truth boxes
        
        Args
        ------
        bb_coord: ground truth bbox coord in original image
        bird_ids: ground truth class labels
        lengths: list containing the number of objects in each img of the batch
        """
        
        batch_size = len(lengths)
        
        # Bbox_overlap, returns a [level]-length list of k (anchors) x n (bbox) array containing corresponding IoUs
        overlaps = [bbox_overlap(anchors_lvl, bb_coord) for anchors_lvl in self.anchors]

        # Most overlapping anchor for each box, such that at least one anchor is assigned per box
        gt_max_overlaps, gt_argmax_overlaps = zip(*[ovlp.max(dim=0) for ovlp in overlaps])
        gt_max_level = torch.stack(gt_max_overlaps).max(dim=0)[1]
        
        # Labels array
        labels = [torch.zeros((batch_size, len(anchors_lvl))).to(self.config.device) for anchors_lvl in self.anchors] # to be converted to one hots
        # Bbox_targets
        reg_targets = [torch.zeros((batch_size, len(anchors_lvl), 4)).to(self.config.device) for anchors_lvl in self.anchors]
        
        indexes = np.cumsum([0] + lengths)
        for b_idx, (i_0, i_f) in enumerate(zip(indexes[:-1], indexes[1:])):
            
            for level in range(len(overlaps)):
                
                indiv_ovlp = overlaps[level][:, i_0:i_f]
                max_overlaps, argmax_overlaps = indiv_ovlp.max(dim=1)
                assigned_ids = bird_ids[i_0:i_f][argmax_overlaps]
                assigned_box = bb_coord[i_0:i_f][argmax_overlaps]
                
                # assign negative labels
                if loss_type == 'focal':
                    ignore_idx = (max_overlaps >= self.config.min_iou_thresh) & (max_overlaps < self.config.max_iou_thresh)
                    max_iou_thresh = self.config.max_iou_thresh
                else:
                    ignore_idx = (max_overlaps < self.config.min_iou_thresh) | \
                        (max_overlaps > self.config.max_iou_thresh) & (max_overlaps < 0.5)
                    max_iou_thresh = 0.5
                labels[level][b_idx, ignore_idx] = -1

                # assign positive class label to the anchor with largest intersection with a gt box
                gt_max_overlap, gt_argmax_overlap = gt_max_overlaps[level][i_0:i_f], gt_argmax_overlaps[level][i_0:i_f]
                is_max_level = gt_max_level[i_0:i_f] == level
                gt_argmax_anchor = gt_argmax_overlap[is_max_level]
                gt_argmax_box = bb_coord[i_0:i_f][is_max_level]
                gt_argmax_id = bird_ids[i_0:i_f][is_max_level]
                # modify assignment (most of the time the anchor should already be assigned to the given box)
                assigned_box[gt_argmax_anchor] = gt_argmax_box
                assigned_ids[gt_argmax_anchor] = gt_argmax_id
                
                # assign positive labels
                pos_idx = (max_overlaps >= max_iou_thresh)
                pos_idx[gt_argmax_anchor] = True
                labels[level][b_idx, pos_idx] = assigned_ids[pos_idx]
                
                # regression targets
                reg_targets[level][b_idx, pos_idx] = bbox_transform(self.anchors[level][pos_idx], assigned_box[pos_idx])

        labels = [label.to(torch.int64) for label in labels]
                
        return labels, reg_targets