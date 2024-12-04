# Copyright (c) NBM. All Rights Reserved

import torch
import torch.nn as nn
from .layers import (RegionProposalNetwork, ProposalLayer,
                     FastRCNN)


class Faster_RCNN(nn.Module):
    
    def __init__(self, args):
        super(Faster_RCNN, self).__init__()
        
        self.args = args
        # RPN
        self.rpn = RegionProposalNetwork(args, args.n_layers, args.top_size)
        # Proposal Layer
        self.prop_layer = ProposalLayer(args, args.n_layers)
        # Fast-RCNN
        self.fast_rcnn = FastRCNN(args)


    def forward(self, x, nms_thresh=0.3, min_score=0.5):
        rois, _, _ = self.forward_first_stage(x)
        return self.forward_second_stage(rois, x, nms_thresh, min_score)


    def forward_second_stage(self, *args, **kwargs):
        return self.fast_rcnn(*args, **kwargs)   


    def forward_first_stage(self, fpn_pyramid_out):
        # Region proposal network
        cls_scores, bbox_reg = self.rpn(fpn_pyramid_out)
        # Proposals - no gradients are expected to flow past this layer
        with torch.no_grad():
            all_rois, _ = self.prop_layer(cls_scores, bbox_reg)
        return all_rois, cls_scores, bbox_reg


def build_head(args):
    return Faster_RCNN(args)