import numpy as np
import torch
import torch.nn as nn
from .util.nets_utils import (weight_init, bbox_reg_to_coord, nms, bbox_overlap, 
                              bbox_transform, get_bbox_regression_targets,
                              generate_anchors_frcnn as generate_anchors,
                              get_anchor_shifts_frcnn as get_anchor_shifts)
from .position_encoding  import one_dimension_positional_encoding
from .self_attention import TransformerEncoder, TransformerEncoderLayer



class DepthwiseSepConv2d(nn.Module):
    """
    Inverted C2d Block
    """

    def __init__(self, indim, outdim, kernel=3, stride=1, expansion_fact=4, bias_out=True, pe_channels=None):
        super().__init__()
        if type(kernel) == tuple:
            padding = (int(0.5 * (kernel[0] - 1)), int(0.5 * (kernel[1] - 1)))
        else:
            padding = int(0.5 * (kernel - 1))
        self.stride = stride
        self.depth_wise = nn.Conv2d(indim, expansion_fact * indim, kernel, stride=int(max(1, stride)), padding=padding,
                                    groups=indim)
        if pe_channels is not None:
            self.pe_proj = nn.Conv2d(pe_channels, 2 * expansion_fact * indim, 1)
        self.pt_wise = nn.Conv2d(expansion_fact * indim, outdim, 1, bias=bias_out)
        self.norm = nn.BatchNorm2d(outdim)
        self.act = nn.SiLU()
        
    def forward(self, x, pe=None):
        # Expected shape: B x C x H x W
        if self.stride < 1:
            size = ((1 / self.stride) * np.array(x.shape[-2:])).astype(np.int64).tolist()
            x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        out = self.depth_wise(x)
        if pe is not None:
            assert hasattr(self, 'pe_proj')
            pe = self.pe_proj(self.act(pe))
            out = out * pe[:, :pe.shape[1] // 2] + pe[:, pe.shape[1] // 2:]
        out = self.pt_wise(out)
        out = self.norm(out)
        out = self.act(out)
        return out


class RegionProposalNetwork(nn.Module):
    
    def __init__(self, args, n_layers, top_layer_size):
        '''
        Params
        n_layers (int): Number of output FPN layers; it is assumed here that the 1st layer is always P1,
            i.e. it corresponds to a 2x decrease of the initial input shape.
        top_layer_size (np.array): Gives the wanted size (h, w) of RPN 1st conv's outputs. This is fixed as 
            the input size (h0, w0) / 2 ** 4.
        '''
        super(RegionProposalNetwork, self).__init__()

        in_cn = args.out_fpn_chan
        self.A = args.n_ratios
        self.convs = nn.ModuleDict({
            str(i): DepthwiseSepConv2d(in_cn, in_cn, stride=(args.anchor_stride / (2 ** (i + 1))), 
                                       expansion_fact=2) for i in range(n_layers)
        })
        self.avgpool = nn.AdaptiveAvgPool2d(top_layer_size)
        self.cls_score = nn.ModuleDict({
            str(i): nn.Conv2d(in_cn, self.A * 2, kernel_size=1) for i in range(n_layers)
        })
        self.bbox_reg = nn.ModuleDict({
            str(i): nn.Conv2d(in_cn, self.A * 4, kernel_size=1) for i in range(n_layers)
        })
        
        # Initialize weights
        self.apply(weight_init)
    
    
    def forward(self, x):
        '''
        x is expected to be the output of a feature pyramid network (a list of feature maps of decreasing resolution)
        '''
        conv_out = [
            self.avgpool(self.convs[str(i)](fm)) for (i, fm) in enumerate(x)
        ]
        # All sizes are identical after the 1st convolutions
        bs, _, h, w = conv_out[0].size()
        # Positive class detection
        cls_scores = [
            self.cls_score[str(i)](fm).view(bs, self.A, 2, h, w).softmax(2) for (i, fm) in enumerate(conv_out)
        ]
        cls_scores = torch.cat(cls_scores, dim=1).view(bs, -1, h, w)
        # Bbox coordinates regression; normally cat() does not interfere with view(), but the latter is still used for clarity purpose
        bbox_reg = [
            self.bbox_reg[str(i)](fm).view(bs, self.A, 4, h, w) for (i, fm) in enumerate(conv_out)
        ]
        bbox_reg = torch.cat(bbox_reg, dim=1).view(bs, -1, h, w)
        
        return cls_scores, bbox_reg
    
    
class AnchorTargetLayer(nn.Module):
        
    def __init__(self, config):
        super(AnchorTargetLayer, self).__init__()
        self.config = config

        height, width = config.top_size
        anchor_stride = config.anchor_stride
        base_size = config.base_size
        ratios = config.ratios
        scales = config.scales

        anchors = generate_anchors(base_size=base_size, ratios=ratios, scales=scales)
        anchors_shifts = get_anchor_shifts(width, height, anchor_stride)
        
        # Move the anchors over the whole image
        all_anchors = (anchors + anchors_shifts).reshape(-1, 4)

        self.A = len(anchors)
        self.K = len(anchors_shifts)
        
        # Keep only anchors inside image
        self.inds_inside = np.where(
            (all_anchors[:, 0] >= 0) & 
            (all_anchors[:, 1] >= 0) & 
            (all_anchors[:, 2] < config.img_width) & 
            (all_anchors[:, 3] < config.img_height))[0]

        self.anchors = torch.Tensor(all_anchors[self.inds_inside]).to(config.device)
        self.all_anchors = all_anchors
    
    
    def forward(self, gt_bbox, lengths):
        """
        Generates regression objectives and classification labels (1 for objects, 0 for bkground, -1 for ignored samples)
        related to each anchor, given the ground truth bbox
        
        Params
        ------
        gt_bbox: ground truth bbox coord in original image
        lengths: list containing the number of objects in each img of the batch
        """
        
        # Params
        batch_size = len(lengths)
        rpn_neg_label = self.config.rpn_neg_label
        rpn_pos_label = self.config.rpn_pos_label
        rpn_batchsize = self.config.rpn_batchsize
        rpn_fg_fraction = self.config.rpn_fg_fraction
        height, width = self.config.top_size
        
        # Bbox_overlap, returns k (anchors) x n (bbox) array containing corresponding IoUs
        overlaps = bbox_overlap(self.anchors, gt_bbox)
        
        # Labels array
        labels = torch.full((batch_size, len(self.inds_inside)), fill_value=-1).to(self.config.device) # one line per batch index
        
        # Bbox_targets
        reg_targets = torch.zeros(batch_size, len(self.anchors), 4).to(self.config.device)
        
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

            reg_targets[b_idx] = bbox_transform(self.anchors, gt_bbox[i_0:i_f][argmax_overlaps])

        # Negative anchors do not participate in regression objective
        reg_targets = labels.unsqueeze(2).clamp(min=0) * reg_targets
        
        # Reshape to original anchor number
        all_labels = torch.full((batch_size, len(self.all_anchors)), fill_value=-1).to(self.config.device)
        all_labels[:, self.inds_inside] = labels

        all_reg_targets = torch.zeros((batch_size, len(self.all_anchors), 4)).to(self.config.device)
        all_reg_targets[:, self.inds_inside] = reg_targets

        all_labels = all_labels.view(-1, height, width, self.A).permute(0, 3, 1, 2)
        all_reg_targets = all_reg_targets.view(-1, height, width, self.A * 4).permute((0, 3, 1, 2))
        
        return all_labels, all_reg_targets
    
    
class ProposalLayer(nn.Module):
        
    def __init__(self, config, n_layers):
        super(ProposalLayer, self).__init__()
        self.n_layers = n_layers
        self.config = config
        
    def forward(self, labels_pred, bbox_reg):
        """
        Takes as input the predicted object scores and bbox regression scores output by the RPN and generates corresponding
        ROIs by shifting base anchors.
        """
        
        batch_size = len(labels_pred)
        
        # Anchors param
        base_size = self.config.base_size
        ratios = self.config.ratios
        scales = 2 ** np.arange(self.n_layers)
        
        anchor_stride = self.config.anchor_stride
        pre_nms_topN = self.config.pre_nms_topN
        min_threshold = self.config.min_threshold
        nms_thresh = self.config.nms_thresh
        post_nms_topN = self.config.post_nms_topN
        if self.training == False:
            post_nms_topN = self.config.post_nms_topN_eval
            pre_nms_topN = self.config.pre_nms_topN_eval
        
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
        # The initial shape of bbox_reg is (batch_size, A * 4, height, width)
        # As for scores, shape is (batch_size, A * 2, height, width) 
        scores = labels_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, K * A, 2)[..., 1]
        bbox_reg = bbox_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, K * A, 4)
        
        # Change from RPN output to absolute coordinates
        
        anchors = torch.Tensor(all_anchors).to(self.config.device)
        bbox_pred = bbox_reg_to_coord(bbox_reg, anchors)
        
        # Clip bbox proposals to image
        
        img_width = self.config.img_width
        img_height = self.config.img_height

        bbox_pred[..., [0, 2]] = bbox_pred[..., [0, 2]].clamp(min=0, max=img_width - 1)
        bbox_pred[..., [1, 3]] = bbox_pred[..., [1, 3]].clamp(min=0, max=img_height - 1)
        
        # Filter out proposals with size < threshold and keep best scoring proposals
        
        keep = (bbox_pred[..., 2] - bbox_pred[..., 0] + 1 >= min_threshold) & \
            ((bbox_pred[..., 3] - bbox_pred[..., 1] + 1 >= min_threshold))

        pre_nms_topN = min(pre_nms_topN, min(keep.sum(dim=1)).item())
        if pre_nms_topN < self.config.rcnn_batch_size:
            print('Not enough possible RoIs, RPN failed')
            return torch.tensor([]).to(self.config.device), torch.tensor([]).to(self.config.device)

        sorted_scores = scores.argsort(descending=True)
        sorted_keep = torch.stack([keep[i, sorted_scores[i]] for i in range(batch_size)])
        pre_nms_idx = torch.stack([sorted_scores[i, sorted_keep[i]][:pre_nms_topN] for i in range(batch_size)])

        scores = torch.stack([scores[i, pre_nms_idx[i]] for i in range(batch_size)])
        bbox_pred = torch.stack([bbox_pred[i, pre_nms_idx[i], :] for i in range(batch_size)])
        
        # Non maximum suppression
        
        bbox_pred, scores = nms(bbox_pred, scores, nms_thresh, post_nms_topN)
        
        return bbox_pred, scores
    
    
class ProposalTargetLayer(nn.Module):
        
    def __init__(self, config):
        super(ProposalTargetLayer, self).__init__()
        self.config = config
        
    def forward(self, rois, gt_bbox, bird_ids, lengths):
        
        # labels = 1 + bird_ids.to(self.config.device) # The object class (bird ID) corresponding to each bbox
        labels = bird_ids.to(self.config.device) # The object class (bird ID) corresponding to each bbox -> the "non-bird sound" class
            # is considered as background class (0)

        num_classes = self.config.num_classes
        
        rcnn_batch_size = self.config.rcnn_batch_size
        rcnn_fg_prop = self.config.rcnn_fg_prop

        fg_threshold = self.config.fg_threshold
        bg_threshold_lo = self.config.bg_threshold_lo
        bg_threshold_hi = self.config.bg_threshold_hi

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
            if len(bg_inds) + len(other_inds) < rcnn_batch_size - fg_rois_per_image:
                n_bgs, n_ign, n_pos = len(bg_inds), len(other_inds), len(fg_inds)
                print(f'~~~~ NOT ENOUGH BG: {n_bgs} / IGNORED ROIS: {n_ign}, FILLING WITH POSITIVES: {n_pos} ~~~~')
                if len(bg_inds) + len(other_inds) < rcnn_batch_size - len(fg_inds):
                    print(f'~~~~ IMPOSSIBLE TO FILL THE RCNN BATCH, NOT ENOUGH ROIS: {n_bgs + n_ign + n_pos} instead of {rcnn_batch_size} ~~~~')
                    return None, None, None
                else:
                    fg_rois_per_image = max(fg_rois_per_image, rcnn_batch_size - (len(bg_inds) + len(other_inds)))
            
            bg_rois_per_image = min(len(bg_inds), rcnn_batch_size - fg_rois_per_image)

            fg_inds = np.random.choice(fg_inds.cpu().numpy(), fg_rois_per_image, replace=False)
            bg_inds = np.random.choice(bg_inds.cpu().numpy(), bg_rois_per_image, replace=False)

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

            out += ((b_rois, bbox_targets, b_labels),)
            
        rois = torch.stack([b_out[0] for b_out in out])
        bbox_targets = torch.stack([b_out[1] for b_out in out])
        labels = torch.stack([b_out[2] for b_out in out])
        
        return rois, bbox_targets, labels
    
    
class ROIPooling(nn.Module):
    """ TODO: To NestedTensors to avoid losing information by squashing the RoIs to a fixed small shape?
    """
    def __init__(self, config):
        super(ROIPooling, self).__init__()
        self.config = config
        
    def forward(self, rois, conv_out):

        def assign_level(box_side_px, max_level):
            return (torch.log(box_side_px * 0.1) / np.log(2)).int().clamp(min=0, max=max_level).cpu().numpy()

        # Each RoI is assigned to a pyramid level depending on its size. Different strides must therefore be used to cast the RoI size from the
        # original image to feature maps in the pyramid levels (lower levels mean smaller stride and larger feature maps)
        heights = [fm.shape[-2] for fm in conv_out]
        widths = [fm.shape[-1] for fm in conv_out]

        rois_size = ((rois[..., 2] - rois[..., 0]) * (rois[..., 3] - rois[..., 1])) ** 0.5
        rois_pyramid_lvl_assignment = assign_level(rois_size, self.config.n_layers - 1)

        stride = torch.Tensor([2 ** (i + 1) for i in range(self.config.n_layers)]).to(self.config.device)
        stride = torch.stack([stride[rois_pyramid_lvl_assignment[i, :]] for i in range(len(rois))])

        roi_pool_h = self.config.roi_pool_h
        roi_pool_w = self.config.roi_pool_w
        # if self.config.rcnn_pe:
        x1 = (rois[..., 0] / stride).round().int()
        y1 = (rois[..., 1] / stride).round().int()
        x2 = (rois[..., 2] / stride).round().int()
        y2 = (rois[..., 3] / stride).round().int()
        # else:# TODO à dégager dès que possible
        #     x1 = (rois[..., 0] / stride).int()
        #     y1 = (rois[..., 1] / stride).int()
        #     x2 = (rois[..., 2] / stride).int()
        #     y2 = (rois[..., 3] / stride).int()

        feature_map_coord = torch.stack([x1, y1, x2, y2], dim=1).transpose(1, 2)

        pe_frequency = one_dimension_positional_encoding(self.config.img_height, self.config.out_fpn_chan // 2).to(rois.device)
        pe_time = one_dimension_positional_encoding(self.config.img_width, self.config.out_fpn_chan // 2).to(rois.device)
        avgpool = nn.AdaptiveAvgPool2d((roi_pool_h, roi_pool_w))

        roi_pool_out, roi_pe_out = [], []

        for batch_idx in range(len(feature_map_coord)):

            b_pool_out, b_roi_pe = [], []

            for i, (x1, y1, x2, y2) in enumerate(feature_map_coord[batch_idx]):
                # Height and width depend on the assigned pyramid level
                fpn_level = rois_pyramid_lvl_assignment[batch_idx, i]
                height = heights[fpn_level]
                width = widths[fpn_level]

                x1 = x1.item()
                x2 = x2.item()
                y1 = y1.item()
                y2 = y2.clamp(max=height - 1).item() # Unorthodox initial image height may create boundary problems.
                # On the contrary, width can be arbitrarily set to a power of 2 so no need to clamp.

                while y2 - y1 + 1 < roi_pool_h:
                    y1 = max(0, y1 - 1)
                    y2 = min(height - 1, y2 + 1)

                while x2 - x1 + 1 < roi_pool_w:
                    x1 = max(0, x1 - 1)
                    x2 = min(width - 1, x2 + 1)
                
                # if not self.config.rcnn_pe:# TODO à dégager dès que possible
                #     roi_h = y2 - y1 + 1
                #     roi_w = x2 - x1 + 1

                #     kernel_height = int(np.ceil(roi_h / roi_pool_h))
                #     kernel_width = int(np.ceil(roi_w / roi_pool_w))

                #     stride_height = int(np.floor(roi_h / roi_pool_h))
                #     stride_width = int(np.floor(roi_w / roi_pool_w))

                #     roi_max_pool = nn.MaxPool2d(kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))
                #     b_pool_out.append(roi_max_pool(conv_out[fpn_level][batch_idx, :, y1: y2 + 1, x1:x2 + 1].unsqueeze(0))[..., :roi_pool_h, :roi_pool_w])
                # else:
                b_pool_out.append(avgpool(conv_out[fpn_level][batch_idx, :, y1: y2 + 1, x1:x2 + 1].unsqueeze(0))) # TODO dégager unsq

                # Positional encoding
                s = stride[batch_idx, i].long().item()
                freq_pe, temp_pe = pe_frequency[s * y1:s * y2], pe_time[:s * (x2 - x1)] # anciennement :s * x2 - x1, à remettre si bug
                roi_pe = torch.cat((
                    freq_pe[:, None] + torch.zeros_like(temp_pe[None]),
                    temp_pe[None] + torch.zeros_like(freq_pe[:, None])
                ), dim=-1).permute(2, 0, 1)
                b_roi_pe.append(avgpool(roi_pe))

            roi_pool_out.append(torch.cat(b_pool_out, dim=0))
            roi_pe_out.append(torch.stack(b_roi_pe))

        roi_pool_out = torch.stack(roi_pool_out)
        roi_pe_out = torch.stack(roi_pe_out)
        
        return roi_pool_out, roi_pe_out, rois_pyramid_lvl_assignment
    
    
class RCNN(nn.Module):
        
    def __init__(self, config):
        super(RCNN, self).__init__()
        
        self.config = config
        roi_pool_h = config.roi_pool_h
        roi_pool_w = config.roi_pool_w
        num_classes = config.num_classes
        hidden_size = config.hidden_size_rcnn
        dropout = config.dropout

        roi_pool_channels = config.out_fpn_chan

        # if config.layered_rcnn:
        #     hidden_size = config.out_fpn_chan * config.roi_pool_h * config.roi_pool_w
        #     self.rcnn = nn.ModuleDict({
        #         str(i): DepthwiseSepConv2d(config.out_fpn_chan, config.out_fpn_chan) for i in range(config.n_layers)
        #     })
        #     self.bbox_reg_layer = nn.ModuleDict({
        #         str(i): nn.Linear(hidden_size, 4 * (1 + num_classes)) for i in range(config.n_layers)
        #     })
        #     self.bbox_classif_layer = nn.ModuleDict({
        #         str(i): nn.Linear(hidden_size, 1 + num_classes) for i in range(config.n_layers)
        #     })
        # elif config.rcnn_pe:
        hidden_size = config.out_fpn_chan * config.roi_pool_h * config.roi_pool_w
        self.pe_proj = nn.Conv2d(config.out_fpn_chan, config.out_fpn_chan, 1)
        self.rcnn = nn.ModuleList([
            DepthwiseSepConv2d(config.out_fpn_chan, config.out_fpn_chan, 
                                pe_channels=config.out_fpn_chan) for _ in range(config.depth_rcnn)
        ])
        self.bbox_reg_layer = nn.Linear(hidden_size, 4 * (1 + num_classes))
        self.bbox_classif_layer = nn.Linear(hidden_size, 1 + num_classes)
        # else:
        #     self.rcnn = nn.Sequential(*[
        #         nn.Linear(roi_pool_channels * roi_pool_h * roi_pool_w, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(p=dropout)
        #     ])

        #     self.bbox_reg_layer = nn.Linear(hidden_size, 4 * (1 + num_classes))
        #     self.bbox_classif_layer = nn.Linear(hidden_size, 1 + num_classes)
        self.softmax_layer = nn.Softmax(dim=-1)

        # Initialize weights
        self.apply(weight_init)

    
    def forward_layer(self, inpt, layer_idx):
        """Input size: [bs (or rather bs * n_rois) x channels x h x w]
        """
        rcnn_out = self.rcnn[str(layer_idx)](inpt).flatten(start_dim=1)
        return self.bbox_reg_layer[str(layer_idx)](rcnn_out), \
                self.softmax_layer(self.bbox_classif_layer[str(layer_idx)](rcnn_out))
        
        
    def forward(self, roi_pool_out, roi_pe_out): #, assigned_level):
        batch_size, rcnn_batch_size, n_channels, roi_pool_h, roi_pool_w = roi_pool_out.shape

        # if self.config.layered_rcnn:
        #     bbox_reg = torch.zeros(batch_size * rcnn_batch_size, 4 * (1 + self.config.num_classes)).to(roi_pool_out.device)
        #     bbox_classes = torch.zeros(batch_size * rcnn_batch_size, 1 + self.config.num_classes).to(roi_pool_out.device)
        #     for i in range(self.config.n_layers):
        #         layer_mask = assigned_level == i
        #         if layer_mask.any():
        #             layer_mask = torch.from_numpy(layer_mask).flatten()[:, None].to(roi_pool_out.device)
        #             lay_reg, lay_cls = self.forward_layer(roi_pool_out.flatten(end_dim=1), i)
        #             bbox_reg = bbox_reg + lay_reg * layer_mask
        #             bbox_classes = bbox_classes + lay_cls * layer_mask
        # elif self.config.rcnn_pe:
        roi_pe = self.pe_proj(roi_pe_out.flatten(end_dim=1))
        out = roi_pool_out.flatten(end_dim=1)
        for depthwise_block in self.rcnn:
            out = depthwise_block(out, roi_pe)
        bbox_reg = self.bbox_reg_layer(out.flatten(start_dim=1))
        bbox_classes = self.softmax_layer(self.bbox_classif_layer(out.flatten(start_dim=1)))
        # else:
        #     out = self.rcnn(roi_pool_out.view(batch_size * rcnn_batch_size, n_channels * roi_pool_h * roi_pool_w))
            
        #     bbox_reg = self.bbox_reg_layer(out)
        #     bbox_classes = self.softmax_layer(self.bbox_classif_layer(out))
        
        return bbox_reg, bbox_classes


class Transformer_RCNN(nn.Module):

    def __init__(self, config):
        super().__init__()

        num_classes = config.num_classes
        in_dim = config.out_fpn_chan * config.roi_pool_h * config.roi_pool_w
        dropout = config.dropout
        model_dim = config.tf_model_dim
        nhead = config.tf_nhead
        num_encoder_layers = config.tf_num_encoder_layers
        dim_feedforward = config.tf_dim_feedforward
        tf_pe_qk = config.tf_pe_qk

        self.pos_embedding = nn.Sequential(*[
            nn.Linear(in_dim, model_dim),
            nn.LeakyReLU()
        ])

        self.rois_embedding = nn.Sequential(*[
            nn.Linear(in_dim, model_dim),
            nn.LeakyReLU()
        ])

        ## Use either the std TF (PE added to the input), or Detr implem (PE only used to compute attn weights)
        if tf_pe_qk:
            encoder_layer = TransformerEncoderLayer(model_dim, nhead, dim_feedforward,
                                                    dropout)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Outputs layers
        self.bbox_reg_layer = nn.Linear(model_dim, 4 * (1 + num_classes))
        self.bbox_classif_layer = nn.Linear(model_dim, 1 + num_classes)
        self.softmax_layer = nn.Softmax(dim=-1)

        self.tf_pe_qk = tf_pe_qk
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, rois, pos):
        # Expected shapes: bs, rcnn_bs, out_fpn_chan, h, w (notice: batch_first is TRUE)

        pos_embed = self.pos_embedding(pos.flatten(start_dim=-3))
        rois_embed = self.rois_embedding(rois.flatten(start_dim=-3))

        if self.tf_pe_qk:
            out = self.encoder(rois_embed, pos_embed)
        else:
            out = self.encoder(rois_embed + pos_embed)
            

        bbox_reg = self.bbox_reg_layer(out).flatten(end_dim=1)
        bbox_classes = self.softmax_layer(self.bbox_classif_layer(out).flatten(end_dim=1))

        return bbox_reg, bbox_classes
    
    
class FastRCNN(nn.Module):
        
    def __init__(self, config):
        super(FastRCNN, self).__init__()
        
        self.config = config
        
        self.roi_pooling = ROIPooling(config)
        if config.tf_rcnn:
            self.rcnn = Transformer_RCNN(config)
        else:
            self.rcnn = RCNN(config)
        
        
    def forward(self, conv_out, rois, nms_thresh=0.3, min_score=0.5, training=None):
        """
        At inference, applies NMS separately for each class to reduce bbox proposal number
        Params:
        rois: RoIs output from ProposalTargetLayer at training time and from ProposalLayer at test time
        conv_out: the pyramid of feature maps output by the FPN
        """
        
        # find the coordinates of each ROI in the last convolutional feature map
        # and process the corresponding sub-frames through ROI pooling layer
        roi_pool_out, roi_pe_out, assigned_level = self.roi_pooling(rois, conv_out)
        # compute the bbox regression values and the class of each ROI
        bbox_reg, bbox_classes = self.rcnn(roi_pool_out, roi_pe_out) # , assigned_level)
        
        if training is None:
            training = self.training
        if training:
            return bbox_reg, bbox_classes
        
        # Inference
        else:
            num_classes = self.config.num_classes
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
                bbox_pred = bbox_reg_to_coord(b_bbox_reg.unsqueeze(0), rois[b_idx])[0]#.squeeze()

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

                    sorted_bbox_pred, sorted_scores, nms_idx = nms(nms_bbox_inpt, nms_scores_inpt, post_nms_topN=len(sorted_bbox_pred), nms_thresh=nms_thresh,
                     return_idx=True)
                    sorted_bbox_pred = sorted_bbox_pred[0]
                    sorted_scores = sorted_scores[0]
                    sorted_classes = sorted_classes[nms_idx]

                # Apply NMS separately for each class
                for class_idx in range(1, num_classes + 1):
                    class_where = torch.nonzero(sorted_classes == class_idx)[:, 0]
                    
                    if len(class_where) == 0:
                        b_output[str(class_idx)] = dict(
                            bbox_coord=torch.Tensor(), 
                            scores=torch.Tensor())
                        continue

                    nms_bbox_inpt = sorted_bbox_pred[class_where].unsqueeze(0)
                    nms_scores_inpt = sorted_scores[class_where].unsqueeze(0)

                    class_bbox_pred, class_scores = nms(nms_bbox_inpt, nms_scores_inpt, post_nms_topN=proposal_number, nms_thresh=nms_thresh)
                    min_scores_idx = torch.nonzero(class_scores > min_score)[:, 1]

                    if len(min_scores_idx) == 0:
                        class_bbox_pred = torch.tensor([])
                        class_scores = torch.tensor([])
                    else:
                        class_bbox_pred = class_bbox_pred.view(class_bbox_pred.size(1), 4)[min_scores_idx]
                        class_scores = class_scores[:, min_scores_idx]

                    b_output[str(class_idx)] = dict(
                        bbox_coord=class_bbox_pred,
                        scores=class_scores
                    )

                output.append(b_output)
        
        return output