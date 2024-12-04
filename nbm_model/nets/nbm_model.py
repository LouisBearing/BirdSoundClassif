# Copyright (c) NBM. All Rights Reserved
# Originates from DETR

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .util import box_ops
from .util.nets_utils import (cross_entropy_loss_rcnn as cross_entropy_loss,
                              smooth_l1_loss_rcnn as smooth_l1_loss,
                              focal_loss_rcnn as focal_loss)

from .backbone import build_backbone
from .fpn import build_fpn
from .self_attention import build_sa_layers
from .head import build_head
from .layers import (AnchorTargetLayer, ProposalTargetLayer)



class NbmModel(nn.Module):
    """ This is the module that performs object detection """
    def __init__(self, args, backbone, attn, fpn, head):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            attn: torch module of the attention layers. See self_attention.py
            fpn: torch module of the FPN layers. See fpn.py
            head: torch module of the head layers (only FasterRCNN implemented at the moment). See head.py
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.attn = attn
        self.fpn = fpn
        self.head = head

    def forward_first_stage(self, samples):
        """ First stage of RoI detection
        Wanted shape: [bs x in_cn x h x w]
        Returns a tensor of RoIs absolute coordinates of shape [bs x n_rois x 4] and the FPN's output pyramid
        """
        features, pos = self.backbone(samples)
        if self.args.add_posenc:
            features = [f + p for (f, p) in zip(features, pos)]
        if self.args.fpn_first:
            fpn_out = self.attn(self.fpn(features))
        elif self.args.sandwich_attn:
            fpn_out = self.attn[1](self.fpn(self.attn[0](features)))
        else:
            fpn_out = self.fpn(self.attn(features))
        pred, cls_scores, bbox_reg = self.head.forward_first_stage(fpn_out)
        return {'rois': pred, 'rpn_cls_scores': cls_scores, 'rpn_bbox_reg': bbox_reg, 'fpn_out': fpn_out}
    
    def forward_second_stage(self, fpn_pyramid_out, rois, nms_thresh=None, min_score=None, training=None):
        if training is None:
            training = self.training
        outputs = self.head.forward_second_stage(fpn_pyramid_out, rois, nms_thresh, min_score, training)
        if training:
            bbox_reg, bbox_classes = outputs
            return {'bbox_reg': bbox_reg, 'bbox_classes': bbox_classes}
        else:
            return outputs

    def forward(self, samples, nms_thresh=0.3, min_score=0.5): # : NestedTensor ?
        """
            Params
               - samples: batched images, of shape [batch_size x 1 x H x W]
               - nms_thresh: max authorized IoU between different boxes
               - min_score: min confidence cutoff for predictions

            It returns a list (of len batch_size) of dictionaries, themselves computes of num_classes dict 
            with the following elements:
               - "scores": the classification confidence for all detected calls of the given species
               - "bbox_coord": The absolute boxes coordinates for all detected calls of the given species
        """

        out = self.forward_first_stage(samples)
        return self.forward_second_stage(out['fpn_out'], out['rois'], nms_thresh, min_score)


class SetCriterion(nn.Module):
    """ This class computes the loss. Right now only FRCNN two-stage loss is implemented.
        Two-stage processes require an intermediate proposal targets computation between 
        stages one and two of the model
    """
    def __init__(self, args, weight_dict):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()

        self.config = args
        self.weight_dict = weight_dict
        # Anchor Target Layer
        self.anchor_target_layer = AnchorTargetLayer(args)
        # Proposal Target Layer
        self.proposal_target_layer = ProposalTargetLayer(args)

    def first_stage_loss(self, labels_pred, bbox_reg, gt_bbox=None, lengths=None, neg_sample=False):
        """
        Computes cross entropy and smooth L1 loss from AnchorTargetLayer objectives
        
        Params
        -----
        labels_pred & bbox_reg are output by the RPN: shapes are (batch_size, A * 2 or 4, height, width)
        gt_bbox, lengths are ground truth boxes and batch lengths (nb of boxes per samples), 
            used to compute labels & reg_targets from AnchorTargetLayer
        """

        if neg_sample:
            # Select most confident predictions
            bs = len(labels_pred)
            labels_pred = labels_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 2)
            _, pos_preds_idx = labels_pred[..., 1].sort(descending=True)
            max_idx = pos_preds_idx[:, :self.config.rcnn_batch_size * 20]
            max_labels_pred = torch.stack([labels_pred[i, max_idx[i]] for i in range(bs)])
            neg_labels = torch.zeros(bs, self.config.rcnn_batch_size * 20, 1).long()
            class_loss = cross_entropy_loss(max_labels_pred, neg_labels, reduction='mean')
            losses = {'first_neg_class_loss': class_loss}
            return losses
        else:
            assert gt_bbox is not None and lengths is not None

        # Compute objective RoIs
        labels, reg_targets = self.anchor_target_layer(gt_bbox, lengths)
        
        # Reshape        
        labels_pred = labels_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        labels = labels.permute(0, 2, 3, 1).flatten().cpu().numpy()
        
        # Filter -1 indices
        keep_idx = np.where(labels != -1)[0]
        labels_pred = labels_pred[keep_idx]
        labels = labels[keep_idx]
        
        # Cross entropy loss
        class_loss = cross_entropy_loss(labels_pred, labels)
        class_loss = class_loss * (1 / len(labels_pred))
        
        # Reshape
        bbox_reg = bbox_reg.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        reg_targets = reg_targets.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        bbox_reg = bbox_reg[keep_idx]
        reg_targets = reg_targets[keep_idx]
        
        # Regression loss
        smoothed_l1 = smooth_l1_loss(bbox_reg, reg_targets)

        # regression loss only applies to the actual object class (or bird species)
        mask = torch.zeros_like(bbox_reg)

        positive_idx = np.where(labels == 1)[0]
        mask[positive_idx, :] = 1

        regression_loss = (mask * smoothed_l1).sum()

        if regression_loss > 0:
            regression_loss = regression_loss * (4 / (labels > 0).sum())
        
        losses = {'first_class_loss': class_loss, 'first_regression_loss': regression_loss}
        return losses
    
    @torch.no_grad()
    def generate_all_rois(self, *args, **kwargs): # proposal_rois, gt_bbox, bird_ids, lengths
        rois, bbox_targets, labels = self.proposal_target_layer(*args, **kwargs)
        return {'rois': rois, 'bbox_targets': bbox_targets, 'labels': labels}

    def second_stage_loss(self, bbox_reg, bbox_classes, bbox_targets=None, labels=None, neg_sample=False):
        """
        Process conv net output through ROI pooling layer and RCNN classifier / regressor, and then computes
        the smooth L1 regression loss and cross entropy classification loss.
        
        Parameters
        ----------
        bbox_reg, bbox_classes are output from FRCNN
        bbox_targets and labels are from from ProposalTargetLayer
        """

        if neg_sample:
            neg_labels = torch.zeros(len(bbox_classes), 1).long()
            class_loss = cross_entropy_loss(bbox_classes, neg_labels, reduction='mean')
            losses = {'sec_neg_class_loss': class_loss}
            return losses
        else:
            assert bbox_targets is not None and labels is not None
        
        batch_size = len(bbox_targets)
        rcnn_batch_size = self.config.rcnn_batch_size
        num_classes = self.config.num_classes
        
        bbox_targets = bbox_targets.view(batch_size * rcnn_batch_size, 4 * (num_classes + 1))
        labels = labels.flatten().cpu().numpy()

        if self.config.focal_loss:
            class_loss = focal_loss(bbox_classes, labels)
        else:
            class_loss = cross_entropy_loss(bbox_classes, labels)
            class_loss = class_loss * (1 / (batch_size * rcnn_batch_size))
        smoothed_l1 = smooth_l1_loss(bbox_reg, bbox_targets)
        
        # regression loss only applies to the actual object class (or bird species)
        mask = torch.zeros_like(bbox_reg)
        for i in range(4):
            mask[range(len(mask)), i + labels * 4] = 1
        # no reg objective for background boxes
        mask[:, 0:4] = 0

        regression_loss = (mask * smoothed_l1).sum()
        if regression_loss > 0:
            regression_loss = regression_loss * (4 / (labels > 0).sum())
        
        losses = {'sec_class_loss': class_loss, 'sec_regression_loss': regression_loss}
        
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        l1 = (outputs.argmax(-1) != 0).sum().item() - (targets != 0).sum().item()
        losses = {'cardinality_error': l1}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes): # TODO -> consider using the GIoU loss from DETR
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, negative_sample=False): # TODO: initial DETR version, rewrite.
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             negative_sample: if True then it's a batch of empty frames, only compute class loss
        """

        device = outputs['pred_logits'].device
        if negative_sample:
            losses = {loss: torch.tensor(0.0).to(device) for loss in ['loss_bbox', 'loss_giou', 'loss_ce', 'cardinality_error']}
            losses['loss_neg_ce'] = (-torch.log(outputs['pred_logits'].softmax(-1)[..., 0])).mean()
            return losses

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        losses['loss_neg_ce'] = torch.tensor(0.0).to(device)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def initialize_model(model, path=None, train=True):

    if path is not None:
        model_dict = model.state_dict()
        exclude = []
        state_dict = torch.load(path)
        state_dict = {k: v for k, v in state_dict['checkpoints'].items() if k in model_dict \
                    and not np.array([e in k for e in exclude]).any()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    if train:
        model.train()
    else:
        model.eval()

    return model


def build(args, train=True):

    device = torch.device(args.device)

    backbone = build_backbone(args) # .train() ?
    if args.fpn_first:
        attn_channels = [args.out_fpn_chan] * len(backbone.num_channels)
    elif args.sandwich_attn:
        attn_channels = (backbone.num_channels, [args.out_fpn_chan] * len(backbone.num_channels))
    else:
        attn_channels = backbone.num_channels
    attn = build_sa_layers(args, attn_channels)
    fpn = build_fpn(args, backbone.num_channels)
    head = build_head(args)
    
    model = NbmModel(
        args,
        backbone,
        attn,
        fpn,
        head
    ).to(device)

    model = initialize_model(model, train=train)

    weight_dict = {
                   'first_class_loss': args.fs_cls_loss_coef,
                   'first_regression_loss': args.fs_reg_loss_coef,
                   'sec_class_loss': args.sec_cls_loss_coef,
                   'sec_regression_loss': args.sec_reg_loss_coef,
                   'first_neg_class_loss': args.fs_neg_cls_loss_coef,
                   'sec_neg_class_loss': args.sec_neg_cls_loss_coef
                }

    criterion = SetCriterion(args, weight_dict)
    criterion.to(device)

    return model, criterion
