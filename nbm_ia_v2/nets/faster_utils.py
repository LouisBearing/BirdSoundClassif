import numpy as np
import torch
import torch.nn as nn

# Utility functions for Faster R-CNN layers

class Config:

    verbose = False
    save_dir = './models'
    conv_net_path = 'checkpoints/vgg16_bn-6c64b313.pth'
    backbone = 'vgg'
    pretrain_conv_net = True
    device = 'cuda'
    classification = False
    
    # General params
    num_classes = 1
    input_channels = 1
    img_width = 1024
    img_height = 256
    use_biophonia = True
    fpn = True
    fpn_rpn = False
    fpn_p_channels = 256
    fpn_o_channels = 256
    normalize_input = False
    zero_max = False
    noise_strength = 0
    self_attention = False
    encode_frequency = False
    position_encoding = False
    transform = False

    # Anchors
    anchor_stride = 16
    base_size = 16
    ratios = [0.5, 1, 2]
    scales_factor_low = 0
    scales_factor_high = 4
    scales = 2**np.arange(scales_factor_low, scales_factor_high)
    n_anchors = len(ratios) * len(scales)
    
    # Anchor Target Layer
    rpn_neg_label = 0.3
    rpn_pos_label = 0.7 # p-e baisser un peu ça
    rpn_batchsize = 16 # jouer là dessus, le réduire exagérément ?
    rpn_fg_fraction = 0.5
    
    # Proposal Layer
    pre_nms_topN = 3000 # jouer là dessus
    min_threshold = 5 # minimum proposal size in px
    nms_thresh = 0.7
    post_nms_topN = 1000 # jouer là dessus
    post_nms_topN_eval = 50
    pre_nms_topN_eval = 500
    
    # Proposal Target Layer
    rcnn_batch_size = 16 # jouer là dessus
    rcnn_fg_prop = 0.4 # 0.25 dans le papier original, essayer différentes valeurs
    fg_threshold = 0.5
    bg_threshold_lo = 0.1
    bg_threshold_hi = 0.5
    
    # ROI Pooling
    roi_pool_h = 2 # à changer (3? 4?)
    roi_pool_w = 2 # à changer (3? 4?)
    hidden_size = 4096
    top_pyramid_roi_size = 128
    rcnn_attention = False
    dropout = 0.5
    
    # Inference
    proposal_number = 50 # number of proposals per class after last nms
    
    # Training
    lambda_reg_rpn_loss = 1.0 # tester qq autres val
    lambda_reg_rcnn_loss = 1.0 # tester qq autres val
    lambda_freq_regul = 0
    batch_size = 2
    val_size = 20
    learning_rate = 1e-4 # jouer la dessus
    validation_prop = 0.01
    n_epochs = 10
    save_every = 10
    scheduler_gamma = 0.1
    scheduler_milestones = [15, 25]
    cv_idx = -1


def generate_anchors(base_size, ratios, scales):
    
    base_anchor_wh = np.array([base_size, base_size])

    # Deform base anchor dimensions to the given ratios 
    coeffs = np.hstack([np.sqrt(ratios)[:, np.newaxis], (1 / np.sqrt(ratios))[:, np.newaxis]])
    ratios_anchors_wh = coeffs * np.sqrt(np.prod(base_anchor_wh))

    # Expand the resulting anchor dimensions to the given sizes
    all_anchor_whs = (ratios_anchors_wh.flatten() * scales[:, np.newaxis]).reshape(-1, 2)

    # Convert from w h to x1 y1 x2 y2 representation, given center coordinates at int(base_size / 2)
    all_anchor = (np.hstack([- all_anchor_whs / 2, all_anchor_whs / 2]) + int(base_size / 2)).astype(int)

    return all_anchor


def get_anchor_shifts(width, height, anchor_stride):

    shift_x = np.arange(0, width) * anchor_stride
    shift_y = np.arange(0, height) * anchor_stride
    shifts = np.hstack([np.tile(shift_x, len(shift_y)).reshape(-1, 1), np.repeat(shift_y, len(shift_x)).reshape(-1, 1)])
    shifts = np.tile(shifts, 2)
    
    return shifts.reshape(-1, 1, 4)


def bbox_overlap(anchors, bbox):
    """
    Computes a K (anchors) x N (bbox) intersection over union matrix
    """
    
    right_boundaries = torch.stack([anchors[:, 2].repeat(len(bbox)), bbox[:, 2].repeat_interleave(len(anchors))]).min(dim=0)[0]
    left_boundaries = torch.stack([anchors[:, 0].repeat(len(bbox)), bbox[:, 0].repeat_interleave(len(anchors))]).max(dim=0)[0]
    x_intersec = (right_boundaries - left_boundaries + 1).clamp(min=0)

    # shapes anchors * bbox

    top_boundaries = torch.stack([anchors[:, 3].repeat(len(bbox)), bbox[:, 3].repeat_interleave(len(anchors))]).min(dim=0)[0]
    bottom_boundaries = torch.stack([anchors[:, 1].repeat(len(bbox)), bbox[:, 1].repeat_interleave(len(anchors))]).max(dim=0)[0]
    y_intersec = (top_boundaries - bottom_boundaries + 1).clamp(min=0)

    intersection = x_intersec * y_intersec

    areas_anchors = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    areas_bbox = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)

    union = torch.stack([areas_anchors.repeat(len(bbox)), areas_bbox.repeat_interleave(len(anchors))]).sum(dim=0) - intersection
    iou = (intersection / union).view(len(bbox), len(anchors)).transpose(1, 0)
    
    return iou


def bbox_transform(anchors, bbox):
    
    wa = (anchors[:, 2] - anchors[:, 0]) + 1
    ha = (anchors[:, 3] - anchors[:, 1]) + 1
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    w = (bbox[:, 2] - bbox[:, 0]) + 1
    h = (bbox[:, 3] - bbox[:, 1]) + 1
    x = bbox[:, 0] + 0.5 * w
    y = bbox[:, 1] + 0.5 * h
    
    t_x = (x - xa) / wa
    t_y = (y - ya) / ha
    t_w = torch.log(w / wa)
    t_h = torch.log(h / ha)
    
    return torch.stack([t_x, t_y, t_w, t_h]).transpose(1, 0)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if (classname.find('Linear') != -1) & (classname.find('LinearLayer') == -1):
        nn.init.kaiming_normal_(m.weight)
    if (classname.find('Conv2d') != -1):
        nn.init.kaiming_normal_(m.weight)


def collate_fn(list_batch):
    lengths = [elt[1].size(0) for elt in list_batch]
    img_batch = torch.stack([img for (img, bb_cord, bird_id, img_info) in list_batch])
    bb_coord_batch = torch.cat([bb_cord for (img, bb_cord, bird_id, img_info) in list_batch], dim=0)
    bird_ids = torch.cat([bird_id for (img, bb_cord, bird_id, img_info) in list_batch])
    img_infos = [elt[-1] for elt in list_batch]
    
    return [img_batch, bb_coord_batch, lengths, bird_ids, img_infos]

        
        
def bbox_reg_to_coord(bbox_pred, anchors):

    wa = (anchors[:, 2] - anchors[:, 0]) + 1
    ha = (anchors[:, 3] - anchors[:, 1]) + 1
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    t_x = bbox_pred[..., 0]
    t_y = bbox_pred[..., 1]
    t_w = bbox_pred[..., 2]
    t_h = bbox_pred[..., 3]

    x = (t_x * wa) + xa
    y = (t_y * ha) + ya
    w = torch.exp(t_w) * wa
    h = torch.exp(t_h) * ha

    return torch.stack([(x - 0.5 * w).round(), (y - 0.5 * h).round(), (x + 0.5 * w).round(), (y + 0.5 * h).round()], dim=2)


def batch_self_overlap(bbox_pred):

    rep = bbox_pred.shape[1]

    right_boundaries = torch.stack([bbox_pred[..., 2].repeat(1, rep), bbox_pred[..., 2].repeat_interleave(rep, dim=1)]).min(dim=0)[0]
    left_boundaries = torch.stack([bbox_pred[..., 0].repeat(1, rep), bbox_pred[..., 0].repeat_interleave(rep, dim=1)]).max(dim=0)[0]
    x_intersec = (right_boundaries - left_boundaries + 1).clamp(min=0)

    top_boundaries = torch.stack([bbox_pred[..., 3].repeat(1, rep), bbox_pred[..., 3].repeat_interleave(rep, dim=1)]).min(dim=0)[0]
    bottom_boundaries = torch.stack([bbox_pred[..., 1].repeat(1, rep), bbox_pred[..., 1].repeat_interleave(rep, dim=1)]).max(dim=0)[0]
    y_intersec = (top_boundaries - bottom_boundaries + 1).clamp(min=0)

    intersection = x_intersec * y_intersec

    areas = (bbox_pred[..., 2] - bbox_pred[..., 0] + 1) * (bbox_pred[..., 3] - bbox_pred[..., 1] + 1)
    union = torch.stack([areas.repeat(1, rep), areas.repeat_interleave(rep, dim=1)]).sum(dim=0) - intersection
    iou = (intersection / union).view(-1, rep, rep)
    
    return iou


def nms(bbox_pred, scores, nms_thresh=0.7, post_nms_topN=300, return_idx=False):
    """
    Applies non maximum suppression to the predicted bbox coordinates bbox_pred (shape batch_size * n_boxes * 4)
    scores are sorted in decreasing order, and bbox_pred coordinates are sorted accordingly for each batch idx
    """
    
    iou = batch_self_overlap(bbox_pred)

    batch_keep = []
    batch_size = len(bbox_pred)

    for b_idx in range(batch_size):

        suppress = []
        keep_idx = []
        b_iou = iou[b_idx]

        for idx in range(len(b_iou)):
            if idx in suppress:
                continue
            keep_idx.append(idx)
            suppress += (torch.nonzero(b_iou[idx, idx + 1:] >= nms_thresh)[:, 0] + idx + 1).tolist()

        batch_keep.append(keep_idx)

    # Truncate idx vectors if one has length < post nms topN
    post_nms_topN = min(np.array([len(b_keep) for b_keep in batch_keep]).min(), post_nms_topN)
    scores = torch.stack([scores[i, batch_keep[i][:post_nms_topN]] for i in range(batch_size)])
    bbox_pred = torch.stack([bbox_pred[i, batch_keep[i][:post_nms_topN], :] for i in range(batch_size)])

    out = bbox_pred, scores

    if return_idx:
        out += (batch_keep,)
    
    return out


def get_bbox_regression_targets(bbox_targets, b_labels, num_classes):
    """
    One regression objective per object class
    """

    expanded_bbox_targets = torch.zeros(len(bbox_targets), 4 * (1 + num_classes)).cuda()
    for i in range(1, num_classes + 1):
        class_idx = torch.nonzero(b_labels == i)[:, 0]
        col_idx = 4 * i
        expanded_bbox_targets[class_idx, col_idx:col_idx + 4] = bbox_targets[class_idx]
        
    return expanded_bbox_targets


def cross_entropy_loss(bbox_classes, labels):
    """
    labels must be a flatten (numpy) array of class indices (0 for background)
    """
    
    gt_probs = bbox_classes[range(len(bbox_classes)), labels]
    cel = (-torch.log(gt_probs)).sum()
    
    return cel


def l1_loss(freq_reg, freq_tgt):
    return torch.abs(freq_reg - freq_tgt).mean()


def smooth_l1_loss(bbox_reg, bbox_targets):

    deltas = torch.abs(bbox_reg - bbox_targets)
    mask_smoothing = (deltas >= 1)
    smoothed_l1 = (~mask_smoothing).float() * 0.5 * (deltas**2) + mask_smoothing.float() * (deltas - 0.5)
    
    return smoothed_l1


def bool_parser(string):
    if string.lower() == 'false':
        return False
    return True


def train_test_split(length, val_prop):
    indices = np.arange(length)
    np.random.shuffle(indices)
    cut = int(val_prop * length)
    return indices[cut:], indices[:cut]

def position_encodings(x, device):
    bs, channels, height, width = x.shape
    i_idx = np.arange(width)
    j_idx = np.arange(height)
    
    position_encodings = torch.from_numpy(np.stack(
        [np.tile(np.sin(i_idx * 128 / (width * (1e4 ** (2 * k / channels)))), (height, 1)) for k in range(int(channels / 4))] + \
        [np.tile(np.cos(i_idx * 128 / (width * (1e4 ** (2 * k / channels)))), (height, 1)) for k in range(int(channels / 4))] + \
        [np.tile(np.sin(j_idx * 128 / (height * (1e4 ** (2 * k / channels))))[:, np.newaxis], (1, width)) for k in range(int(channels / 4))] + \
        [np.tile(np.cos(j_idx * 128 / (height * (1e4 ** (2 * k / channels))))[:, np.newaxis], (1, width)) for k in range(int(channels / 4))]
    )).to(device)
    
    return position_encodings.unsqueeze(0).repeat(bs, 1, 1, 1).float()