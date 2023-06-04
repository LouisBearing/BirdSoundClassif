import os
import numpy as np
from .faster_utils import *
import torch
import torch.nn as nn


class RegionProposalNetwork(nn.Module):
    
    def __init__(self, config, in_channels=512, out_channels=512):
        super(RegionProposalNetwork, self).__init__()
        
        self.fpn = config.fpn_rpn
        
        if not self.fpn:
            self.A = config.n_anchors
            self.conv = nn.Conv2d(in_channels + 6 * int(config.encode_frequency), out_channels, padding=1, kernel_size=3)
        else:
            # Makes sur there's exactly 4 different scales, one for each feature pyramid layer
            assert len(config.scales) == 4
            self.A = int(config.n_anchors / 4)
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=1, kernel_size=3, stride=iter_stride) for iter_stride in [8, 4, 2, 1]
            ])
        self.activation = nn.ReLU()

        self.cls_score = nn.Conv2d(out_channels, self.A * 2, kernel_size=1)
        self.softmax_layer = nn.Softmax(dim=2)
        self.bbox_reg = nn.Conv2d(out_channels, self.A * 4, kernel_size=1)
        
        # Initialize weights
        self.apply(weight_init)
    
    
    def forward(self, x):

        if not self.fpn:
            conv_out = self.activation(self.conv(x))
            batch_size, n_channels, height, width = conv_out.shape
            cls_scores = self.cls_score(conv_out)
            cls_scores = self.softmax_layer(cls_scores.view(batch_size, self.A, 2, height, width)).view(batch_size, -1, height, width)
            bbox_reg = self.bbox_reg(conv_out)

        # if FPN, x is a list of bottom-up (decreasing size, increasing semantic meaning) activations output by the conv net backbone
        else:
            conv_out = [self.activation(rpn_conv(fm)) for rpn_conv, fm in zip(self.conv, x)]
            batch_size, n_channels, height, width = conv_out[0].shape # all have same shape now
            # Objectness scores
            cls_scores = [self.cls_score(x) for x in conv_out]
            cls_scores = [self.softmax_layer(x.view(batch_size, self.A, 2, height, width)) for x in cls_scores]
            # Predictions are reversed to match the ordering of generate_anchors: from smaller to larger anchors (or top-down flow in the case of FPN)
            cls_scores = torch.cat([_ for _ in cls_scores[::-1]], dim=1).view(batch_size, -1, height, width)
            # Bbox coordinates regression
            bbox_reg = [self.bbox_reg(x).view(batch_size, self.A, 4, height, width) for x in conv_out]
            # Same thing here
            bbox_reg = torch.cat([_ for _ in bbox_reg[::-1]], dim=1).view(batch_size, -1, height, width)
        
        return cls_scores, bbox_reg
    
    
    def compute_losses(self, labels_pred, bbox_reg, labels, reg_targets):
        """
        Computes cross entropy and smooth L1 loss from AnchorTargetLayer objectives
        
        Params
        -----
        labels_pred & bbox_reg are output by the RPN: shapes are (batch_size, A * 2 or 4, height, width)
        labels & reg_targets are regression and classification output from AnchorTargetLayer
        """
        
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
        
        return class_loss, regression_loss
    
    
class AnchorTargetLayer(nn.Module):
        
    def __init__(self):
        super(AnchorTargetLayer, self).__init__()
    
    
    def forward(self, gt_bbox, lengths, config, height, width):
        """
        Generates regression objectives and classification labels (1 for objects, 0 for bkground, -1 for ignored samples)
        related to each anchor, given the ground truth bbox
        
        Params
        ------
        gt_bbox: ground truth bbox coord in original image
        lengths: list containing the number of objects in each img of the batch
        height & width correspond to the dimensions of the last feature map of the classifier backbone (usually 16 * 32)
        """
        
        # Params
        batch_size = len(lengths)
        anchor_stride = config.anchor_stride

        base_size = config.base_size
        ratios = config.ratios
        scales = config.scales

        rpn_neg_label = config.rpn_neg_label
        rpn_pos_label = config.rpn_pos_label
        rpn_batchsize = config.rpn_batchsize
        rpn_fg_fraction = config.rpn_fg_fraction

        anchors = generate_anchors(base_size=base_size, ratios=ratios, scales=scales)
        anchors_shifts = get_anchor_shifts(width, height, anchor_stride)
        
        # Move the anchors over the whole image
        all_anchors = (anchors + anchors_shifts).reshape(-1, 4)

        A = len(anchors)
        K = len(anchors_shifts)
        
        # Keep only anchors inside image
        img_width = anchor_stride * width
        img_height = anchor_stride * height

        inds_inside = np.where(
            (all_anchors[:, 0] >= 0) & 
            (all_anchors[:, 1] >= 0) & 
            (all_anchors[:, 2] < img_width) & 
            (all_anchors[:, 3] < img_height))[0]

        anchors = torch.Tensor(all_anchors[inds_inside]).to(config.device)
        
        # Bbox_overlap, returns k (anchors) x n (bbox) array containing corresponding IoUs
        overlaps = bbox_overlap(anchors, gt_bbox)
        
        # Labels array
        labels = torch.full((batch_size, len(inds_inside)), fill_value=-1).to(config.device) # one line per batch index
        
        # Bbox_targets
        reg_targets = torch.zeros(batch_size, len(anchors), 4).to(config.device)
        
        indexes = np.cumsum([0] + lengths)
        for b_idx, (i_0, i_f) in enumerate(zip(indexes[:-1], indexes[1:])):
            indiv_ovlp = overlaps[:, i_0:i_f]
            max_overlaps, argmax_overlaps = indiv_ovlp.max(dim=1)
            gt_max_overlaps, gt_argmax_overlaps = indiv_ovlp.max(dim=0)

            # assign negative labels

            labels[b_idx, max_overlaps < rpn_neg_label] = 0

            # assign positive labels

            labels[b_idx, max_overlaps >= rpn_pos_label] = 1
            # only if the max overlap with any bbox is greater than zero are the corresponding anchors assigned a positive label
            if gt_max_overlaps.max().item() > 0:
                positive_idx = torch.nonzero(gt_max_overlaps > 0)[:, 0]
                gt_argmax_overlaps = torch.nonzero(indiv_ovlp[:, positive_idx] == gt_max_overlaps[positive_idx])[:, 0]
                labels[b_idx, gt_argmax_overlaps] = 1

            # subsample positive anchors

            num_fg = int(rpn_fg_fraction * rpn_batchsize)
            fg_inds = torch.nonzero(labels[b_idx] == 1)[:, 0]

            if len(fg_inds) > num_fg:
                disable_idx = np.random.choice(fg_inds.cpu().numpy(), len(fg_inds) - num_fg, replace=False)
                labels[b_idx, disable_idx] = -1

            # subsample negative anchors    

            num_bg = rpn_batchsize - len(torch.nonzero(labels[b_idx] == 1)[:, 0])
            bg_inds = torch.nonzero(labels[b_idx] == 0)[:, 0]

            if len(bg_inds) > num_bg:
                disable_idx = np.random.choice(bg_inds.cpu().numpy(), len(bg_inds) - num_bg, replace=False)
                labels[b_idx, disable_idx] = -1

            # regression targets

            reg_targets[b_idx] = bbox_transform(anchors, gt_bbox[i_0:i_f][argmax_overlaps])

        # Negative anchors do not participate in regression objective
        reg_targets = labels.unsqueeze(2).clamp(min=0) * reg_targets
        
        # Reshape to original anchor number
        all_labels = torch.full((batch_size, len(all_anchors)), fill_value=-1).to(config.device)
        all_labels[:, inds_inside] = labels

        all_reg_targets = torch.zeros((batch_size, len(all_anchors), 4)).to(config.device)
        all_reg_targets[:, inds_inside] = reg_targets

        all_labels = all_labels.view(-1, height, width, A).permute(0, 3, 1, 2)
        all_reg_targets = all_reg_targets.view(-1, height, width, A * 4).permute((0, 3, 1, 2))
        
        return all_labels, all_reg_targets
    
    
class ProposalLayer(nn.Module):
        
    def __init__(self):
        super(ProposalLayer, self).__init__()
        
    def forward(self, labels_pred, bbox_reg, config, training=True):
        """
        Takes as input the predicted object scores and bbox regression scores output by the RPN and generates corresponding
        ROIs by shifting base anchors.
        """
        
        batch_size = len(labels_pred)
        
        # Anchors param
        base_size = config.base_size
        ratios = config.ratios
        scales = config.scales
        
        anchor_stride = config.anchor_stride
        pre_nms_topN = config.pre_nms_topN
        min_threshold = config.min_threshold
        nms_thresh = config.nms_thresh
        post_nms_topN = config.post_nms_topN
        if training == False:
            post_nms_topN = config.post_nms_topN_eval
            pre_nms_topN = config.pre_nms_topN_eval
        
        height, width = labels_pred.shape[-2:]
        
        # Generate base anchors and shifts
        
        anchors = generate_anchors(base_size=base_size, ratios=ratios, scales=scales)
        anchors_shifts = get_anchor_shifts(width, height, anchor_stride)

        # Move the anchors over the whole image
        
        # final anchors shape: (K * A, 4) where K is itself height * width
        all_anchors = (anchors + anchors_shifts).reshape(-1, 4)

        A = len(anchors)
        K = len(anchors_shifts)
                        
        # Reshape
        # The initial shape of bbox_reg is (batch_size, A * 4, height, width) with usually height = 16 and width = 32
        # As for scores, shape is (batch_size, A * 2, height, width) 
        scores = labels_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, K * A, 2)[..., 1]
        bbox_reg = bbox_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, K * A, 4)
        
        # Change from RPN output to absolute coordinates
        
        anchors = torch.Tensor(all_anchors).to(config.device)
        bbox_pred = bbox_reg_to_coord(bbox_reg, anchors)
        
        # Clip bbox proposals to image
        
        img_width = config.img_width
        img_height = config.img_height

        bbox_pred[..., [0, 2]] = bbox_pred[..., [0, 2]].clamp(min=0, max=img_width - 1)
        bbox_pred[..., [1, 3]] = bbox_pred[..., [1, 3]].clamp(min=0, max=img_height - 1)
        
        # Filter out proposals with size < threshold and keep best scoring proposals
        
        keep = (bbox_pred[..., 2] - bbox_pred[..., 0] + 1 >= min_threshold) & \
            ((bbox_pred[..., 3] - bbox_pred[..., 1] + 1 >= min_threshold))

        pre_nms_topN = min(pre_nms_topN, min(keep.sum(dim=1)).item())
        # if pre_nms_topN == 0:
        #     return torch.tensor([]).cuda(), torch.tensor([]).cuda()

        sorted_scores = scores.argsort(descending=True)
        sorted_keep = torch.stack([keep[i, sorted_scores[i]] for i in range(batch_size)])
        pre_nms_idx = torch.stack([sorted_scores[i, sorted_keep[i]][:pre_nms_topN] for i in range(batch_size)])

        scores = torch.stack([scores[i, pre_nms_idx[i]] for i in range(batch_size)])
        bbox_pred = torch.stack([bbox_pred[i, pre_nms_idx[i], :] for i in range(batch_size)])
        
        # Non maximum suppression
        
        bbox_pred, scores = nms(bbox_pred, scores, nms_thresh, post_nms_topN)
        
        return bbox_pred, scores
    
    
class ProposalTargetLayer(nn.Module):
        
    def __init__(self):
        super(ProposalTargetLayer, self).__init__()
        
    def forward(self, rois, gt_bbox, bird_ids, lengths, config):
        
        labels = 1 + bird_ids.to(config.device) # The object class (bird ID) corresponding to each bbox

        if not config.classification:
            labels = labels.bool().int()

        num_classes = config.num_classes
        
        rcnn_batch_size = config.rcnn_batch_size
        rcnn_fg_prop = config.rcnn_fg_prop

        fg_threshold = config.fg_threshold
        bg_threshold_lo = config.bg_threshold_lo
        bg_threshold_hi = config.bg_threshold_hi

        assert bg_threshold_hi <= fg_threshold
        
        out = tuple()

        indexes = np.cumsum([0] + lengths)
        for b_idx, (i_0, i_f) in enumerate(zip(indexes[:-1], indexes[1:])):

            gt_boxes = gt_bbox[i_0:i_f]

            # Add gt boxes to roi list if they concern positive samples
            if gt_boxes.max().item() > -1:
                all_rois = torch.cat([rois[b_idx], gt_boxes], dim=0)
            else:
                all_rois = rois[b_idx]

            # Bbox_overlap, returns n (bbox) x k (anchors) array containing corresponding IoUs
            overlaps = bbox_overlap(all_rois, gt_boxes)
            max_overlap, assignment = overlaps.max(dim=-1)

            # Labels of each proposal's assigned bbox
            b_labels = labels[i_0:i_f][assignment]
            b_labels[max_overlap < fg_threshold] = 0

            # Assigned gt boxes
            gt_boxes = gt_boxes[assignment]

            # Subsample foreground and background rois
            fg_inds = torch.nonzero(max_overlap > fg_threshold)[:, 0]
            bg_inds = torch.nonzero((bg_threshold_hi > max_overlap) & (max_overlap >= bg_threshold_lo))[:, 0]
            other_inds = list(set(range(len(max_overlap))) - set(bg_inds.cpu().numpy()) - set(fg_inds.cpu().numpy()))

            fg_rois_per_image = min(len(fg_inds), int(rcnn_fg_prop * rcnn_batch_size))
            bg_rois_per_image = min(len(bg_inds), rcnn_batch_size - fg_rois_per_image)

            fg_inds = np.random.choice(fg_inds.cpu().numpy(), fg_rois_per_image, replace=False)
            bg_inds = np.random.choice(bg_inds.cpu().numpy(),bg_rois_per_image, replace=False)

            if len(fg_inds) + len(bg_inds) < rcnn_batch_size:
                bg_inds = np.hstack([
                    bg_inds,
                    np.random.choice(other_inds, rcnn_batch_size - len(fg_inds) - len(bg_inds), replace=False)
                ])

            # Slice labels and rois with the rcnn_batch_size kept indices
            keep_inds = np.hstack((fg_inds, bg_inds))
            b_labels = b_labels[keep_inds]
            b_rois = all_rois[keep_inds]

            # Targets
            bbox_targets = bbox_transform(b_rois, gt_boxes[keep_inds])

            # Reshape from rcnn_batch_size * 4 to rcnn_batch_size * (4 * num_classes)
            bbox_targets = get_bbox_regression_targets(bbox_targets, b_labels, num_classes)

            # Frequency regression targets
            b_freq_targets = (b_rois[:, [1]] + 0.5 * (b_rois[:, [3]] - b_rois[:, [1]])) / config.img_height

            out += ((b_rois, bbox_targets, b_labels, b_freq_targets),)
            
        rois = torch.stack([b_out[0] for b_out in out])
        bbox_targets = torch.stack([b_out[1] for b_out in out])
        labels = torch.stack([b_out[2] for b_out in out])
        freq_targets = torch.stack([b_out[3] for b_out in out])
        
        return rois, bbox_targets, labels, freq_targets
    
    
class ROIPooling(nn.Module):
        
    def __init__(self):
        super(ROIPooling, self).__init__()
        
    def forward(self, rois, conv_out, config):

        fpn = config.fpn

        if fpn:
            # Each RoI is assigned to a pyramid level depending on its size. Different strides must therefore be used to cast the RoI size from the
            # original image to feature maps in the pyramid levels (lower levels mean smaller stride and larger feature maps)
            heights = [fm.shape[-2] for fm in conv_out]
            widths = [fm.shape[-1] for fm in conv_out]

            rois_size = ((rois[..., 2] - rois[..., 0]) * (rois[..., 3] - rois[..., 1])) ** 0.5
            rois_pyramid_lvl_assignment = (3 + torch.log(rois_size / config.top_pyramid_roi_size)).int().clamp(min=0, max=3).cpu().numpy()

            if config.backbone == 'vgg':
                stride = torch.Tensor([2, 4, 8, 16]).to(config.device)
            else:
                stride = torch.Tensor([4, 8, 16, 32]).to(config.device)
            stride = torch.stack([stride[rois_pyramid_lvl_assignment[i, :]] for i in range(len(rois))])

        else:
            height, width = conv_out.shape[-2:]
            stride = config.anchor_stride

        roi_pool_h = config.roi_pool_h
        roi_pool_w = config.roi_pool_w

        x1 = (rois[..., 0] / stride).int()
        y1 = (rois[..., 1] / stride).int()
        x2 = (rois[..., 2] / stride).int()
        y2 = (rois[..., 3] / stride).int()
        feature_map_coord = torch.stack([x1, y1, x2, y2], dim=1).transpose(1, 2)

        roi_pool_out = []

        for batch_idx in range(len(feature_map_coord)):

            b_pool_out = []

            for i, (x1, y1, x2, y2) in enumerate(feature_map_coord[batch_idx]):

                if fpn:
                    # Height and width depend on the assigned pyramid level
                    fpn_level = rois_pyramid_lvl_assignment[batch_idx, i]
                    height = heights[fpn_level]
                    width = widths[fpn_level]

                x1 = x1.item()
                x2 = x2.item()
                y1 = y1.item()
                y2 = y2.item()

                while y2 - y1 + 1 < roi_pool_h:
                    y1 = max(0, y1 - 1)
                    y2 = min(height - 1, y2 + 1)

                while x2 - x1 + 1 < roi_pool_w:
                    x1 = max(0, x1 - 1)
                    x2 = min(width - 1, x2 + 1)

                roi_h = y2 - y1 + 1
                roi_w = x2 - x1 + 1

                kernel_height = int(np.ceil(roi_h / roi_pool_h))
                kernel_width = int(np.ceil(roi_w / roi_pool_w))

                stride_height = int(np.floor(roi_h / roi_pool_h))
                stride_width = int(np.floor(roi_w / roi_pool_w))

                roi_max_pool = nn.MaxPool2d(kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))

                if fpn:
                    b_pool_out.append(roi_max_pool(conv_out[fpn_level][batch_idx, :, y1: y2 + 1, x1:x2 + 1].unsqueeze(0))[..., :roi_pool_h, :roi_pool_w])
                else:
                    b_pool_out.append(roi_max_pool(conv_out[batch_idx, :, y1: y2 + 1, x1:x2 + 1].unsqueeze(0))[..., :roi_pool_h, :roi_pool_w])

            roi_pool_out.append(torch.cat(b_pool_out, dim=0))

        roi_pool_out = torch.stack(roi_pool_out)
        
        return roi_pool_out
    
    
class RCNN(nn.Module):
        
    def __init__(self, config):
        super(RCNN, self).__init__()
        
        roi_pool_h = config.roi_pool_h
        roi_pool_w = config.roi_pool_w
        num_classes = config.num_classes
        hidden_size = config.hidden_size
        dropout = config.dropout

        if config.fpn:
            roi_pool_channels = config.fpn_o_channels
        else:
            roi_pool_channels = 512 + int(config.encode_frequency)

        self.rcnn_attention = config.rcnn_attention
        if self.rcnn_attention:
            self.attention_module = SelfAttention(roi_pool_channels, roi_pool_channels)

        self.rcnn = nn.Sequential(*[
            LinearLayer(roi_pool_channels * roi_pool_h * roi_pool_w, hidden_size, activation='relu', norm='none', dropout=dropout),
            LinearLayer(hidden_size, hidden_size, activation='relu', norm='none', dropout=dropout)
        ]).to(config.device)

        self.bbox_reg_layer = LinearLayer(hidden_size, 4 * (1 + num_classes)).to(config.device)
        self.bbox_classif_layer = LinearLayer(hidden_size, 1 + num_classes).to(config.device)
        self.softmax_layer = nn.Softmax(dim=-1)

        # Frequency regression net
        # self.freq_reg_layer = LinearLayer(hidden_size, 1).to(config.device)
        
        # Initialize weights
        self.apply(weight_init)
        
        
    def forward(self, roi_pool_out):
        
        batch_size, rcnn_batch_size, n_channels, roi_pool_h, roi_pool_w = roi_pool_out.shape
        if self.rcnn_attention:
            roi_pool_out = roi_pool_out.view(batch_size * rcnn_batch_size, n_channels, roi_pool_h * roi_pool_w)
            roi_pool_out += self.attention_module(roi_pool_out)
        out = self.rcnn(roi_pool_out.view(batch_size * rcnn_batch_size, n_channels * roi_pool_h * roi_pool_w))
        
        bbox_reg = self.bbox_reg_layer(out)
        # freq_reg = self.freq_reg_layer(out)
        freq_reg = 0
        bbox_classes = self.softmax_layer(self.bbox_classif_layer(out))
        
        return bbox_reg, bbox_classes, freq_reg
    
    
class FastRCNN(nn.Module):
        
    def __init__(self, config):
        super(FastRCNN, self).__init__()
        
        self.config = config
        
        self.roi_pooling = ROIPooling()
        self.rcnn = RCNN(config)
        
        
    def compute_losses(self, rois, bbox_targets, labels, frequencies, conv_out):
        """
        Process conv net output through ROI pooling layer and RCNN classifier / regressor, and then computes
        the smooth L1 regression loss and cross entropy classification loss.
        
        Parameters
        ----------
        rois, bbox_targets and labels are from from ProposalTargetLayer
        conv_out is output by VGG / ResNet backbone
        """
        
        batch_size = len(bbox_targets)
        rcnn_batch_size = self.config.rcnn_batch_size
        num_classes = self.config.num_classes
        
        bbox_targets = bbox_targets.view(batch_size * rcnn_batch_size, 4 * (num_classes + 1))
        labels = labels.flatten().cpu().numpy()
        frequencies = frequencies.flatten(end_dim=1)
        
        bbox_reg, bbox_classes, freq_reg = self.forward(rois, conv_out)

        class_loss = cross_entropy_loss(bbox_classes, labels)
        class_loss = class_loss * (1 / (batch_size * rcnn_batch_size))
        freq_reg_loss = l1_loss(freq_reg, frequencies)
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
        
        return class_loss, regression_loss, freq_reg_loss
        
        
    def forward(self, rois, conv_out, training=True, inter_nms_thresh=0.3, intra_nms_thresh=0.3, min_score=0.5):
        """
        At inference, applies NMS separately for each class to reduce bbox proposal number
        rois comes from ProposalTargetLayer at training time and from ProposalLayer at test time
        """
        
        # find the coordinates of each ROI in the last convolutional feature map
        # and process the corresponding sub-frames through ROI pooling layer
        roi_pool_out = self.roi_pooling(rois, conv_out, self.config)
        # compute the bbox regression values and the class of each ROI
        bbox_reg, bbox_classes, freq_reg = self.rcnn(roi_pool_out)
        
        if training:
            return bbox_reg, bbox_classes, freq_reg
        
        # Inference
        else:
            
            num_classes = self.config.num_classes
            # If it's classification, add a -1 offset to output class
            if num_classes == 1:
                offset = 0
            else:
                offset = -1
            batch_size, rcnn_batch_size = rois.shape[:2]
            proposal_number = self.config.proposal_number

            scores, predicted_class = bbox_classes.max(dim=1)
            
            # Extract only the regression values that correspond to the predicted class
            new_line_indices = np.arange(len(bbox_reg)) * 4 * (num_classes + 1)
            indices = (predicted_class.cpu().numpy() * 4) + new_line_indices
            indices = np.repeat(indices, 4) + np.tile(np.arange(4), len(indices))
            bbox_reg = bbox_reg.flatten()[indices].view(batch_size, rcnn_batch_size, 4)

            # bbox_reg = bbox_reg[:, 4:].view(batch_size, rcnn_batch_size, 4)
            # scores = bbox_classes[:, 1].view(batch_size, rcnn_batch_size)
            
            # Reshape
            predicted_class = predicted_class.view(batch_size, -1)
            scores = scores.view(batch_size, -1)
            sorted_idx = scores.argsort(descending=True)

            output = []
            
            # Iterate batch and append final results
            for b_idx in range(batch_size):

                b_output = {}

                b_bbox_reg = bbox_reg[b_idx]

                # Apply regressions to ROIs
                bbox_pred = bbox_reg_to_coord(b_bbox_reg.unsqueeze(0), rois[b_idx]).squeeze()

                # Clip bbox proposals to image
                img_width = self.config.img_width
                img_height = self.config.img_height

                bbox_pred[:, [0, 2]] = bbox_pred[:, [0, 2]].clamp(min=0, max=img_width - 1)
                bbox_pred[:, [1, 3]] = bbox_pred[:, [1, 3]].clamp(min=0, max=img_height - 1)

                # Sort according to decreasing confidence
                sorted_scores = scores[b_idx, sorted_idx[b_idx]]
                sorted_bbox_pred = bbox_pred[sorted_idx[b_idx]]
                sorted_classes = predicted_class[b_idx, sorted_idx[b_idx]]

                # First NMS, all classes + suppress class 0
                non_zeros_where = torch.nonzero(sorted_classes > 0)[:, 0]

                if len(non_zeros_where) > 0:

                    nms_bbox_inpt = sorted_bbox_pred[non_zeros_where].unsqueeze(0)
                    nms_scores_inpt = sorted_scores[non_zeros_where].unsqueeze(0)
                    sorted_classes = sorted_classes[non_zeros_where]

                    sorted_bbox_pred, sorted_scores, nms_idx = nms(nms_bbox_inpt, nms_scores_inpt, post_nms_topN=len(sorted_bbox_pred), nms_thresh=inter_nms_thresh,
                     return_idx=True)
                    sorted_bbox_pred = sorted_bbox_pred[0]
                    sorted_scores = sorted_scores[0]
                    sorted_classes = sorted_classes[nms_idx]

                # Apply NMS separately for each class
                for class_idx in range(1, num_classes + 1):
                    class_where = torch.nonzero(sorted_classes == class_idx)[:, 0]
                    
                    if len(class_where) == 0:
                        b_output[str(class_idx + offset)] = dict(
                            bbox_coord=torch.Tensor(), 
                            scores=torch.Tensor())
                        continue

                    nms_bbox_inpt = sorted_bbox_pred[class_where].unsqueeze(0)
                    nms_scores_inpt = sorted_scores[class_where].unsqueeze(0)

                    class_bbox_pred, class_scores = nms(nms_bbox_inpt, nms_scores_inpt, post_nms_topN=proposal_number, nms_thresh=intra_nms_thresh)
                    min_scores_idx = torch.nonzero(class_scores > min_score)[:, 1]

                    if len(min_scores_idx) == 0:
                        class_bbox_pred = torch.tensor([])
                        class_scores = torch.tensor([])
                    else:
                        class_bbox_pred = class_bbox_pred.view(class_bbox_pred.size(1), 4)[min_scores_idx]
                        class_scores = class_scores[:, min_scores_idx]

                    b_output[str(class_idx + offset)] = dict(
                        bbox_coord=class_bbox_pred,
                        scores=class_scores
                    )

                # class_idx = 1
                # b_output[str(class_idx)] = dict(
                #     bbox_coord=sorted_bbox_pred, 
                #     scores=sorted_scores
                # )

                output.append(b_output)
        
        return output


class LinearLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation='none', norm='none', dropout=0):
        super(LinearLayer, self).__init__()

        self.linear = nn.utils.spectral_norm(nn.Linear(in_channels, out_channels)) 
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
            
        if norm == 'in':
            self.norm = nn.InstanceNorm1d(out_channels)
        elif norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels, momentum=0.5)
        else:
            self.norm = None
            
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        
    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            if self.norm.__class__.__name__.find('InstanceNorm') > -1:
                x = self.norm(x.unsqueeze(1)).squeeze()
            else:
                x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, input_dim, inner_dim):
        super(SelfAttention, self).__init__()
        
        self.query = LinearLayer(input_dim, inner_dim)
        self.key = LinearLayer(input_dim, inner_dim)
        self.value = LinearLayer(input_dim, inner_dim)
        self.final_projection = LinearLayer(inner_dim, input_dim)


    def forward(self, x, position_encoding=False):

        bs, input_dim, height, width = x.size()
        L = height * width

        x = x.flatten(start_dim=-2)
        x = x.transpose(1, 2).contiguous().flatten(end_dim=-2)
        
        queries = self.query(x).view(bs, L, -1)
        keys = self.key(x).view(bs, L, -1)
        values = self.value(x).view(bs, L, -1)

        if position_encoding:

            queries = queries.transpose(1, 2).view(bs, -1, height, width)
            keys = keys.transpose(1, 2).view(bs, -1, height, width)

            pos_encoding = position_encodings(queries, queries.device)
            queries = (queries + pos_encoding).flatten(start_dim=-2).transpose(1, 2)
            keys = (keys + pos_encoding).flatten(start_dim=-2).transpose(1, 2)
        
        factors = torch.softmax(torch.matmul(queries, keys.transpose(1, 2)) / np.round(np.sqrt(queries.size(-1)), 2), dim=-1)
        context_vect = torch.matmul(factors, values)
        context_vect = self.final_projection(context_vect.flatten(end_dim=-2)).view(bs, L, input_dim).transpose(1, 2).contiguous()

        return context_vect