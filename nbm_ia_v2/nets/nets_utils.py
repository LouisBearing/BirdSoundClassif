import numpy as np
import torch
import torch.nn as nn

SIZES = [(94, 256), (47, 128), (24, 64), (12, 32), (6, 16)]
CHANNELS = [48, 80, 160, 176, 512]
BCKB_LAYERS = [2, 3, 4, 5, 7]
N_ANCHORS = 9
IMG_SIZE = (375, 1024)

class Config:

    save_dir = './models'
    device = 'cuda'
    batch_size = 5
    
    # General params
    n_classes = 144

    # Anchors
    base_size = 16
    ratios = [0.5, 1, 2]
    min_level = 3
    max_level = 7
    
    # Anchor Target Layer
    min_iou_thresh = 0.4
    max_iou_thresh = 0.5

    # Detector
    dropout = 0.1
    n_layers_bifpn = 7 # 2
    out_channels = 224 # 32,
    n_layers_branches = 4 # 1,
    min_score = 0.5
    inter_nms_thresh = 0.3
    intra_nms_thresh = 0.3

    # Focal loss
    gamma = 1.5
    alpha = 0.25
    
    # Training
    learning_rate = 1e-4 #
    validation_prop = 0.03
    n_epochs = 30
    scheduler_gamma = 0.1
    scheduler_milestones = [15, 25]


def generate_anchors(base_size, ratios, n_scales):
    
    base_anchor_wh = np.array([base_size, base_size])

    # Deform base anchor dimensions to the given ratios 
    coeffs = np.hstack([np.sqrt(ratios)[:, np.newaxis], (1 / np.sqrt(ratios))[:, np.newaxis]])
    additional_ratios = np.array([2 ** (i / 3) for i in np.arange(3)])
    additional_ratios = additional_ratios[:, np.newaxis, np.newaxis]
    coeffs = (additional_ratios * coeffs[np.newaxis, ...]).reshape(-1, 2)
    ratios_anchors_wh = coeffs * np.sqrt(np.prod(base_anchor_wh))

    # Expand the resulting anchor dimensions to the given sizes
    all_anchor_whs = [(ratios_anchors_wh.flatten() * (2 ** i)).reshape(-1, 2) for i in np.arange(n_scales)]

    # Convert from w h to x1 y1 x2 y2 representation, given center coordinates at int(base_size / 2)
    all_anchors = [(np.hstack([-anchor_wh / 2, anchor_wh / 2]) + int(base_size / 2)).astype(int) for anchor_wh in all_anchor_whs]

    return all_anchors


def get_anchor_shifts_level(size, level):

    height, width = size
    stride = 2 ** (level - 1)

    shift_x = np.arange(width) * stride
    shift_y = np.arange(height) * stride
    shifts = np.hstack([np.tile(shift_x, len(shift_y)).reshape(-1, 1), np.repeat(shift_y, len(shift_x)).reshape(-1, 1)])
    shifts = np.tile(shifts, 2)
    
    return shifts.reshape(-1, 1, 4)


def get_anchor_shifts(min_level, max_level):
    shifts = []
    for i in range(min_level, max_level + 1):
        size = SIZES[i - min_level]
        shifts.append(get_anchor_shifts_level(size, i))
    return shifts


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
    lengths = [len(elt[2]) for elt in list_batch]
    img_batch = torch.stack([img for (img, neg_img, bb_coord, bird_id) in list_batch])
    neg_img_batch = torch.stack([neg_img for (img, neg_img, bb_coord, bird_id) in list_batch])
    bb_coord_batch = torch.cat([bb_coord for (img, neg_img, bb_coord, bird_id) in list_batch], dim=0)
    bird_ids = torch.cat([bird_id for (img, neg_img, bb_coord, bird_id) in list_batch])
    
    return [img_batch, neg_img_batch, bb_coord_batch, bird_ids, lengths]

        
        
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


def smooth_l1_loss(bbox_reg, bbox_targets, pos_idx):
    """
    Expected input shape: (bs, n_anchors, 4) or (bs, n_anchors) for pos_idx
    """

    deltas = torch.abs(bbox_reg - bbox_targets)
    mask_smoothing = (deltas >= 1)
    smoothed_l1 = (~mask_smoothing).float() * 0.5 * (deltas**2) + mask_smoothing.float() * (deltas - 0.5)
    smoothed_l1 = smoothed_l1.sum(dim=-1)

    norm_weights = pos_idx.sum(dim=1)
    
    return (smoothed_l1 * pos_idx.float()).sum(dim=1) / norm_weights


def focal_loss(gt_class_logits, ignore_idx, pos_idx, alpha, gamma):
    """
    Args:
    -----
    gt_class_logits (bs, n_anchors): the logits corresponding to the gt class of the assigned boxes
    ignore_idx (bs, n_anchors): ignored anchors
    pos_idx (bs, n_anchors): foreground assigned class
    """
    loss = -(1 - gt_class_logits).pow(gamma) * torch.log(gt_class_logits)
    loss = (1.0 - ignore_idx) * loss
    loss = torch.where(pos_idx, alpha * loss, (1.0 - alpha) * loss)
    norm_weights = pos_idx.sum(dim=1)

    return loss.sum(dim=1) / norm_weights


def focal_loss_neg(backgnd_logits, gamma):
    """
    Args:
    -----
    backgnd_logits (bs, n_anchors): the logits corresponding to the background class for each (negative) anchor
    """
    loss = -(1 - backgnd_logits).pow(gamma) * torch.log(backgnd_logits)

    return loss.sum(dim=1)


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