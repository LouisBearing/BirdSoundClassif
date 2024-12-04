# Copyright (c) NBM. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pickle
import glob
import os
import itertools


IMG_SIZE = (375, 1024)

# SA_y, SA_minx, SA_maxx = 768, 60, 90

class Config:

    save_dir = './models'
    device = 'cuda'
    batch_size = 8
    

def coord_to_rel(bb_coord):
    h, w = bb_coord[:, 3] - bb_coord[:, 1], bb_coord[:, 2] - bb_coord[:, 0]
    x0, y0 = bb_coord[:, 0] + 0.5 * w, bb_coord[:, 1] + 0.5 * h
    return torch.stack([x0 / IMG_SIZE[1], y0 / IMG_SIZE[0], w / IMG_SIZE[1], h / IMG_SIZE[0]], dim=1)

def rel_to_coord(bb_coord_rel):
    x0, y0, w, h = bb_coord_rel[:, 0] * IMG_SIZE[1], bb_coord_rel[:, 1] * IMG_SIZE[0], \
        bb_coord_rel[:, 2] * IMG_SIZE[1], bb_coord_rel[:, 3] * IMG_SIZE[0]
    return torch.stack([x0 - 0.5 * w, y0 - 0.5 * h, x0 + 0.5 * w, y0 + 0.5 * h], dim=1).round()


def generate_anchors_frcnn(base_size, ratios, scales):
    
    base_anchor_wh = np.array([base_size, base_size])

    # Deform base anchor dimensions to the given ratios 
    coeffs = np.hstack([np.sqrt(ratios)[:, np.newaxis], (1 / np.sqrt(ratios))[:, np.newaxis]])
    ratios_anchors_wh = coeffs * np.sqrt(np.prod(base_anchor_wh))

    # Expand the resulting anchor dimensions to the given sizes
    all_anchor_whs = (ratios_anchors_wh.flatten() * scales[:, np.newaxis]).reshape(-1, 2)

    # Convert from w h to x1 y1 x2 y2 representation, given center coordinates at int(base_size / 2)
    all_anchor = (np.hstack([- all_anchor_whs / 2, all_anchor_whs / 2]) + int(base_size / 2)).astype(int)

    return all_anchor


def get_anchor_shifts_frcnn(width, height, anchor_stride):

    shift_x = np.arange(0, width) * anchor_stride
    shift_y = np.arange(0, height) * anchor_stride
    shifts = np.hstack([np.tile(shift_x, len(shift_y)).reshape(-1, 1), np.repeat(shift_y, len(shift_x)).reshape(-1, 1)])
    shifts = np.tile(shifts, 2)
    
    return shifts.reshape(-1, 1, 4)


def generate_anchors(base_size, ratios, n_scales):
    
    base_anchor_wh = np.array([base_size, base_size])

    # Deform base anchor dimensions to the given ratios 
    coeffs = np.hstack([np.sqrt(ratios)[:, np.newaxis], (1 / np.sqrt(ratios))[:, np.newaxis]])
    # additional_ratios = np.array([2 ** (i / 3) for i in np.arange(3)])
    additional_ratios = np.array([0.75, 1, 1.2])
    additional_ratios = additional_ratios[:, np.newaxis, np.newaxis]
    coeffs = (additional_ratios * coeffs[np.newaxis, ...]).reshape(-1, 2)
    ratios_anchors_wh = coeffs * np.sqrt(np.prod(base_anchor_wh))

    # Expand the resulting anchor dimensions to the given sizes
    all_anchor_whs = [(ratios_anchors_wh.flatten() * (2 ** i)).reshape(-1, 2) for i in np.arange(n_scales)]

    # Convert from w h to x1 y1 x2 y2 representation, given center coordinates at int(base_size / 2)
    all_anchors = [np.hstack([-anchor_wh / 2, anchor_wh / 2]).astype(int) for anchor_wh in all_anchor_whs]

    return all_anchors


def get_anchor_shifts_level(size):

    height, width = size
    stride_y, stride_x = ((IMG_SIZE[0] - 1) / (size[0] - 1), (IMG_SIZE[1] - 1) / (size[1] - 1))

    shift_x = (np.arange(width) * stride_x).astype(int)
    shift_y = (np.arange(height) * stride_y).round(0).astype(int)
    shifts = np.hstack([np.tile(shift_x, len(shift_y)).reshape(-1, 1), np.repeat(shift_y, len(shift_x)).reshape(-1, 1)])
    shifts = np.tile(shifts, 2)
    
    return shifts.reshape(-1, 1, 4)


def get_anchor_shifts(bckb):
    shifts = []
    for size in SIZES[bckb]:
        shifts.append(get_anchor_shifts_level(size))
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
    if (classname.find('Conv2d') != -1) and (classname.find('DepthwiseSepConv2d') == -1):
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
    Applies non maximum suppression to the predicted bbox coordinates bbox_pred (shape batch_size x n_boxes x 4)
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

    expanded_bbox_targets = torch.zeros(len(bbox_targets), 4 * (1 + num_classes)).to(bbox_targets.device)
    for i in range(1, num_classes + 1):
        class_idx = torch.nonzero(b_labels == i)[:, 0]
        col_idx = 4 * i
        expanded_bbox_targets[class_idx, col_idx:col_idx + 4] = bbox_targets[class_idx]
        
    return expanded_bbox_targets


def cross_entropy_loss_rcnn(bbox_classes, labels, reduction='sum'):
    """
    labels must be a flatten (numpy) array of class indices (0 for background)
    """
    
    gt_probs = bbox_classes[range(len(bbox_classes)), labels]
    cel = -torch.log(gt_probs)
    if reduction == 'sum':
        return cel.sum()
    elif reduction == 'mean':
        return cel.mean()


def smooth_l1_loss_rcnn(bbox_reg, bbox_targets):

    deltas = torch.abs(bbox_reg - bbox_targets)
    mask_smoothing = (deltas >= 1)
    smoothed_l1 = (~mask_smoothing).float() * 0.5 * (deltas**2) + mask_smoothing.float() * (deltas - 0.5)
    
    return smoothed_l1


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


def scheduler_fn_alpha(step, max_step=25000, log_a_bar_init=-5, a_bar_max=0.5):
    
    log_a_bar_max = np.log10(a_bar_max)
    log_a_bar = np.clip((log_a_bar_init + step * (log_a_bar_max - log_a_bar_init) / max_step), a_min=None, a_max=log_a_bar_max)
    a = 1 - 10 ** log_a_bar
    return a


def cross_entropy_loss(gt_class_logits, ignore_idx, n_boxes):

    loss = -torch.log(gt_class_logits)
    loss = (1.0 - ignore_idx) * loss

    return loss.sum(dim=1) / n_boxes


def focal_loss_rcnn(bbox_classes, labels, reduction='mean', gamma=1.5):
    """
    labels must be a flatten (numpy) array of class indices (0 for background)
    """
    gt_class_logits = bbox_classes[range(len(bbox_classes)), labels]
    factor = (1 - gt_class_logits).pow(gamma)
    loss = -factor * torch.log(gt_class_logits)
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()


def focal_loss(gt_class_logits, ignore_idx, pos_idx, alpha, gamma):
    """
    Args:
    -----
    gt_class_logits (bs, n_anchors): the logits corresponding to the gt class of the assigned boxes
    ignore_idx (bs, n_anchors): ignored anchors
    pos_idx (bs, n_anchors): foreground assigned class
    """
    factor = (1 - gt_class_logits).pow(gamma)
    loss = -factor * torch.log(gt_class_logits)
    loss = (1.0 - ignore_idx) * loss
    loss = torch.where(pos_idx, alpha * loss, (1.0 - alpha) * loss)
    norm_weights = pos_idx.sum(dim=1)

    return loss.sum(dim=1) / norm_weights


def focal_loss_neg(backgnd_logits, gamma, alpha):
    """
    Args:
    -----
    backgnd_logits (bs, n_anchors): the logits corresponding to the background class for each (negative) anchor
    """
    factor = (1 - backgnd_logits).pow(gamma)
    loss = -factor * torch.log(backgnd_logits)

    return (1.0 - alpha) * loss.sum(dim=1)


def bool_parser(string):
    if string.lower() == 'false':
        return False
    return True


def train_test_split(length, val_prop):
    indices = np.arange(length)
    np.random.shuffle(indices)
    cut = int(val_prop * length)
    return indices[cut:], indices[:cut]


def extract_samples_from_labels(labels, n_samples=48):
    ignore_idx = torch.ones_like(labels).to(torch.float)
    
    pos_idx = labels > 0
    bg_idx = labels == 0
    neg_idx = labels == -1
    
    n_pos = pos_idx.sum(dim=1)
    for b_idx in range(len(n_pos)):
        wanted_pos = min(n_pos[b_idx].item(), int(n_samples / 2))
        b_pos_idx = torch.where(pos_idx[b_idx])[0].cpu().numpy()
        if len(b_pos_idx) > 0:
            b_pos_idx = np.random.choice(b_pos_idx, wanted_pos, replace=False)

        wanted_bg = int(n_samples / 2)
        b_bg_idx = torch.where(bg_idx[b_idx])[0].cpu().numpy()
        if len(torch.where(bg_idx[b_idx])[0].cpu().numpy()) > 0:
            b_bg_idx = np.random.choice(b_bg_idx, wanted_bg, replace=False)
        idx_list = np.hstack([b_pos_idx, b_bg_idx])

        b_missing = n_samples - len(b_pos_idx) - len(b_bg_idx)
        if b_missing > 0:
            idx_list = np.hstack([
                idx_list, np.random.choice(torch.where(neg_idx[b_idx])[0].cpu().numpy(), b_missing, replace=False)
            ])
        
        ignore_idx[b_idx, idx_list] = 0
    
    return ignore_idx


def setattr_others(args):
    if args.n_ratios == 3:
        setattr(args, 'ratios', [0.5, 1, 2])
    elif args.n_ratios == 5:
        setattr(args, 'ratios', [0.2, 0.5, 1, 2, 5])
    if 'vgg' in args.backbone:
        setattr(args, 'n_layers', 4)
        setattr(args, 'top_size', (23, 64))
    else:
        setattr(args, 'n_layers', 5)
        setattr(args, 'top_size', (24, 64))
    setattr(args, 'scales', 2 ** np.arange(args.n_layers))


def read_annot_file(annot_path):
    annots = []
    with open(annot_path, 'r') as f:
        annot = []
        for i, line in enumerate(f):
            if i % 2 == 0:
                annot = []
                annot.append(line)
            else:
                annot.append(line)
                annots.append(annot)
    return annots


def format_single_annot(annot, pix_precision_y=33.3, pix_precision_x=0.002993197278911565, low_freq=500, h_pix=375):
    t0, t1, spec = annot[0].replace('\n', '').split('\t')
    f0, f1 = annot[1].replace('\n', '').replace('\\\t', '').split('\t')
    return (
        spec, [np.round(float(t0) / pix_precision_x), np.round((float(f0) - low_freq) / pix_precision_y).clip(min=0), 
               np.round(float(t1) / pix_precision_x), np.round((float(f1) - low_freq) / pix_precision_y).clip(max=h_pix - 1)]
    )


def format_txt_annots(annot_path):
    annots = read_annot_file(annot_path)
    formated_annots = {}
    for annot in annots:
        spec, coords = format_single_annot(annot)
        if spec in formated_annots.keys():
            formated_annots[spec].append(coords)
        else:
            formated_annots[spec] = [coords]
    return formated_annots


def compute_AP_scores(outputs, filter_sp=None):
    """Computes the AP (all species) and the mAP (averaged over species-dependent AP scores) from output scores.
    Params
        outputs (list): list of tuples, containing the outputs of the models and the (formatted) GT boxes, 
            both represented as directories with bird species as keys.
    Returns
        AP - Average precision over all predicted boxes of all (test) files
        mAP - Mean AP, ie the former but grouped by species and averaged over the latters.
        Rec, mRec - Same with recall. To calculate the recall, all boxes are considered provided the IoU is above threshold, whatever the 
            confidence score.
    """
    # Format the output list into a friendly pandas DataFrame
    formatted_scores = pd.DataFrame()
    for i, (output, formatted_annots) in enumerate(outputs):
        fp = [spec for spec in output.keys() if spec not in formatted_annots.keys()]
        fn = [spec for spec in formatted_annots.keys() if spec not in output.keys()]
        tp = [spec for spec in output.keys() if spec in formatted_annots.keys()]
    
        for spec in tp:
            iou = bbox_overlap(torch.tensor(output[spec]['bbox_coord']), torch.tensor(formatted_annots[spec])).max(dim=1)[0]
            formatted_scores = pd.concat(
                [formatted_scores, pd.DataFrame({'file_idx': i, 'species': spec, 'iou': iou.numpy(), 'scores': output[spec]['scores']})],
                ignore_index=True
            )
        for spec in fp:
            formatted_scores = pd.concat(
                [formatted_scores, pd.DataFrame({'file_idx': i, 'species': spec, 'iou': 0.0, 'scores': output[spec]['scores']})],
                ignore_index=True
            )
        for spec in fn:
            formatted_scores = pd.concat(
                [formatted_scores, pd.DataFrame({'file_idx': i, 'species': len(formatted_annots[spec]) * [spec], 'iou': 0.0, 'scores': 0.0})],
                ignore_index=True
            )
    if len(formatted_scores) == 0:
        return {'AP': 0, 'mAP': 0, 'Rec': 0, 'mRec': 0}
    
    # IoU threshold for TP is hardcoded at 0.5
    formatted_scores['type'] = 'FP'
    formatted_scores.loc[formatted_scores['iou'] >= 0.5, 'type'] = 'TP'
    formatted_scores.loc[formatted_scores['scores'] == 0, 'type'] = 'FN'
    
    # Sort by decreasing confidence
    formatted_scores = formatted_scores.sort_values('scores', ascending=False)

    if filter_sp is not None:
        formatted_scores = formatted_scores.loc[formatted_scores['species'].map(lambda x: x in filter_sp)]

    AP, Rec = calculate_ap(formatted_scores)

    mscores = []
    for spec, df in formatted_scores.groupby('species'):
        mscores.append((spec, calculate_ap(df)))
    mAP = np.array([s[0] for _, s in mscores if s[0] > -1]).mean()
    mRec = np.array([s[1] for _, s in mscores]).mean()
    
    return {'AP': AP, 'mAP': mAP, 'Rec': Rec, 'mRec': mRec}


def calculate_ap(pandas_df):
    count = pandas_df.value_counts('type')
    for key in ['TP', 'FP', 'FN']:
        if not key in count.keys():
            count[key] = 0
    recall = count['TP'] / max(1, count['TP'] + count['FN'])
    if count['TP'] + count['FP'] == 0:
        return -1, recall
    pandas_df['precision'] = (pandas_df['type'] == 'TP').astype(int).cumsum().div(np.arange(1, len(pandas_df) + 1).clip(max=count['TP'] + count['FP']))
    pandas_df['recall'] = (pandas_df['type'] == 'TP').astype(int).cumsum().div(max(1, count['TP'] + count['FN']))

    # Precision interpolation
    interp = pandas_df.groupby('recall').agg({'precision': 'max'}).reset_index().rename(columns={'precision': 'prec_interp'})
    pandas_df = pandas_df.merge(interp, on='recall')
    # Categorize recall
    pandas_df['recall_bins'] = pd.cut(pandas_df['recall'], bins=np.arange(0, 1.1, 0.1), 
                                                 include_lowest=True, labels=np.arange(10))
    # Take AP as the mean precision within each recall bin, then over bins. Check that this is the standard way.
    # AP = pandas_df.groupby('recall_bins', observed=True)[['prec_interp']].apply(lambda x: x.mean())['prec_interp'].mean()
    bin_means = pandas_df.groupby('recall_bins', observed=True)[['prec_interp']].apply(lambda x: x.mean())['prec_interp']
    AP = bin_means.values.sum() / 10
    return AP, recall


def calculate_recall(recall_matrix):
    count = recall_matrix.value_counts('type')
    for key in ['TP', 'FN']:
        if not key in count.keys():
            count[key] = 0
    recall = count['TP'] / max(1, count['TP'] + count['FN'])
    return recall


def calculate_mScore(mat, score, min_n_boxes=0, filter_sp=None):
    mscores = []
    if filter_sp is not None:
        mat = mat.loc[mat['species'].map(lambda x: x in filter_sp)].copy()
    for spec, df in mat.groupby('species'):
        if len(df) > min_n_boxes:
            if score == 'ap':
                mscores.append((spec, calculate_ap(df)))
            elif score == 'recall':
                mscores.append((spec, calculate_recall(df)))
    if score == 'ap':
        return mscores, np.array([s[0] for _, s in mscores if s[0] > -1]).mean()
    elif score == 'recall':
        return np.array([s for _, s in mscores if s > -1]).mean()


# def execute_birdnet(test_dir, out_path='birdnet_testset_results'): --> does not work, model not found?
#     # Avoid unnecessary imports at file reading 
#     import time
#     from pathlib import Path
#     from birdnet.models import ModelV2M4

#     # create model instance for v2.4
#     model = ModelV2M4()
#     outputs = []
#     t = time.time()
#     for i, wav_path in enumerate(glob.glob(test_dir + '/*.wav')):
#         predictions = model.predict_species_within_audio_file(Path(wav_path), min_confidence=0.02, batch_size=5)
#         outputs.append((os.path.basename(wav_path), predictions))
#         print(f'i: {i}, t: {time.time() - t}')
#     with open(out_path, 'wb') as f:
#         pickle.dump(outputs, f)


def find_windows(left, right, delta, win_size=3.0):
    return [(win_size * i, win_size * (i + 1)) \
        for i in range((int(left + delta) // int(win_size)), int((right - delta) // int(win_size)) + 1)]


def format_model_output_df(model_outputs, src, delta_px=5):
    pix_precision_x = 0.002993197278911565

    if src == 'birdnet':
        formatted_outputs = pd.DataFrame()
        for (file, out) in model_outputs:
            form_out = [[(np.round(float(t[0]) / pix_precision_x), np.round(float(t[1]) / pix_precision_x), s.split('_')[0], c) \
                        for (s, c) in win_out_dict.items()] for (t, win_out_dict) in out.items()]
            form_out = list(itertools.chain.from_iterable(form_out))
            pd_form_out = pd.DataFrame(form_out).rename(columns={0: 't_0', 1: 't_f', 2: 'species', 3: 'scores'})
            pd_form_out['file_idx'] = file.replace('.wav', '')
            formatted_outputs = pd.concat(
                        [formatted_outputs, pd_form_out],
                        ignore_index=True
                    )

    elif src == 'nbm':
        delta = delta_px * pix_precision_x
        formatted_outputs = pd.DataFrame()
        for (file, out) in model_outputs:
            file_res_flat = [[(sp, sp_info['bbox_coord'][i][0] * pix_precision_x, sp_info['bbox_coord'][i][2] * pix_precision_x, sp_info['scores'][i]) \
                        for i in range(len(sp_info['scores']))] for sp, sp_info in out.items()]
            file_res_flat = list(itertools.chain.from_iterable(file_res_flat))
            file_res_flat = [[(s, np.round(float(l) / pix_precision_x), np.round(float(r) / pix_precision_x), c) for (l, r) in find_windows(t0, t1, delta)] \
                            for (s, t0, t1, c) in file_res_flat]
            file_res_flat = list(itertools.chain.from_iterable(file_res_flat))
            pd_file_res = pd.DataFrame(file_res_flat).rename(columns={0: 'species', 1: 't_0', 2: 't_f', 3: 'scores'})
            pd_file_res['file_idx'] = file.replace('.wav', '')
            formatted_outputs = pd.concat(
                        [formatted_outputs, pd_file_res],
                        ignore_index=True
                    )

        formatted_outputs = formatted_outputs.sort_values('scores', ascending=False).drop_duplicates(['species', 't_0', 'file_idx'])
    
    elif src == 'birdnetlib':
        formatted_outputs = pd.DataFrame()
        for (file, out) in model_outputs:
            pd_form_out = pd.DataFrame(out).rename(columns={'start_time': 't_0', 'end_time': 't_f', 'common_name': 'file_idx', 
            'confidence': 'scores', 'label': 'species'})
            for key in ['t_0', 't_f']:
                pd_form_out[key] = pd_form_out[key].map(lambda x: np.round(float(x) / pix_precision_x))
            pd_form_out['file_idx'] = file.replace('.wav', '')
            formatted_outputs = pd.concat(
                        [formatted_outputs, pd_form_out],
                        ignore_index=True
                    )
        formatted_outputs['species'] = formatted_outputs['species'].str.split('_', expand=True)[0]

    return formatted_outputs


def format_annotations_df(annotations, delta_px=5):
    pix_precision_x = 0.002993197278911565
    delta = delta_px * pix_precision_x
    formatted_annotations = pd.DataFrame()
    for (file, out) in annotations:
        form_out = [[(s, coord[0], coord[2]) for coord in box] for (s, box) in out.items()]
        form_out = list(itertools.chain.from_iterable(form_out))
        form_out_flat = [[(s, np.round(float(l) / pix_precision_x), np.round(float(r) / pix_precision_x)) \
            for (l, r) in find_windows(t0 * pix_precision_x, t1 * pix_precision_x, delta)] \
                for (s, t0, t1) in form_out]
        form_out_flat = list(itertools.chain.from_iterable(form_out_flat))
        pd_form_out = pd.DataFrame(form_out_flat).rename(columns={0: 'species', 1: 't_0', 2: 't_f'})
        pd_form_out['file_idx'] = file.replace('.wav', '')
        formatted_annotations = pd.concat(
                    [formatted_annotations, pd_form_out],
                    ignore_index=True
                )
        formatted_annotations = formatted_annotations.drop_duplicates()

    return formatted_annotations


def compute_metrics_sliding_windows_out(test_dir, out_path, src='birdnet', delta_px=5):
    with open(out_path, 'rb') as f:
        model_outputs = pickle.load(f)
    annotations = []
    for wav_path in glob.glob(test_dir + '/*.wav'):
        annotations.append((os.path.basename(wav_path), format_txt_annots(wav_path.replace('.wav', '.txt'))))
    tgt_species = list(np.unique([a[0].split('#')[0].capitalize().replace('_', ' ') for a in annotations]))
    rem_sp = ['Anas platyrhynchos', 'Anthus campestris', 'Luscinia megarhynchos'] # Not enough training data for these species in current version (09/2024)
    tgt_species = [e for e in tgt_species if e not in rem_sp]

    ## Format sliding window model outputs
    formatted_outputs = format_model_output_df(model_outputs, src, delta_px)
    ## Format annotations
    formatted_annotations = format_annotations_df(annotations)
    # Merge
    formatted_scores = pd.merge(formatted_outputs, formatted_annotations, on=['file_idx', 'species'], how='outer', 
                            suffixes=['_out', '_annot'])
    # Calculate intersection between annotation boxes and output 3s windows, then sort and split for prec / rec calculation
    formatted_scores['intersection'] = [min(tf_1, tf_2) - max(t0_1, t0_2) for (t0_1, tf_1, t0_2, tf_2) in zip(
        formatted_scores['t_0_out'], formatted_scores['t_f_out'], formatted_scores['t_0_annot'], formatted_scores['t_f_annot']
    )]
    formatted_scores.loc[[(a or b) for (a,b) in zip(formatted_scores['t_f_out'].isnull(), formatted_scores['t_f_annot'].isnull())],
    'intersection'] = np.nan
    formatted_scores.sort_values(by='intersection', ascending=False, inplace=True)

    # Filter out files from excluded species -> TODO: suppress whenever not needed anymore
    formatted_scores = formatted_scores.loc[~formatted_scores['file_idx'].map(
        lambda x: x.split('#')[0].capitalize().replace('_', ' ') in rem_sp
        )]

    # Recall matrix
    recall_matrix = formatted_scores.loc[~formatted_scores['t_0_annot'].isnull()].drop_duplicates(['file_idx', 'species', 't_0_annot', 't_f_annot'])
    recall_matrix['type'] = 'FN'
    recall_matrix.loc[recall_matrix['intersection'] > delta_px, 'type'] = 'TP'
    recall = calculate_recall(recall_matrix)
    mRec = calculate_mScore(recall_matrix, 'recall', 5)
    # Prepare the concatenation with precision matrix by setting scores to zeros so that FNs does not interfere in AP calculation
    # recall_matrix['scores'] = 0

    # Precision matrix
    precision_matrix = formatted_scores.loc[~formatted_scores['t_0_out'].isnull()].drop_duplicates(['file_idx', 'species', 't_0_out'])
    precision_matrix['type'] = 'FP'
    precision_matrix.loc[precision_matrix['intersection'] > delta_px, 'type'] = 'TP'
    precision_matrix.sort_values(by='scores', ascending=False, inplace=True)
    precision_matrix = pd.concat([precision_matrix, recall_matrix.loc[recall_matrix['type'] == 'FN']], ignore_index=True)
    AP, _ = calculate_ap(precision_matrix)
    mAP = calculate_mScore(precision_matrix, 'ap', filter_sp=tgt_species)

    return AP, recall, mAP, mRec, precision_matrix