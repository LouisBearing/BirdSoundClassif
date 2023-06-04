import os
import argparse
import numpy as np
import json

from nets.faster_rcnn import *
from nets.faster_utils import *
from nets.layers import *
from nets.vgg_backbone import *
from nets.resnet_backbone import *
from pytorch_dataset.image_dataset import *
from torch.utils.data.sampler import SubsetRandomSampler


config = Config()
parser = argparse.ArgumentParser()

### General params
parser.add_argument('--classification', default=config.classification, type=bool_parser)
parser.add_argument('--verbose', default=config.verbose, type=bool_parser)
parser.add_argument('--model_name', default='new_model', type=str)
parser.add_argument('--data_path', default=os.getcwd(), type=str)
parser.add_argument('--save_dir', default=config.save_dir, type=str)
parser.add_argument('--pretrain_conv_net', default=config.pretrain_conv_net, type=bool_parser)
parser.add_argument('--backbone', default=config.backbone, type=str)
parser.add_argument('--conv_net_path', default=config.conv_net_path, type=str)
parser.add_argument('--fpn_rpn', default=config.fpn_rpn, type=bool_parser)
parser.add_argument('--fpn', default=config.fpn, type=bool_parser)
parser.add_argument('--fpn_p_channels', default=config.fpn_p_channels, type=int)
parser.add_argument('--fpn_o_channels', default=config.fpn_o_channels, type=int)
parser.add_argument('--use_biophonia', default=config.use_biophonia, type=bool_parser)
parser.add_argument('--img_width', default=config.img_width, type=int)
parser.add_argument('--normalize_input', default=config.normalize_input, type=bool_parser)
parser.add_argument('--zero_max', default=config.zero_max, type=bool_parser)
parser.add_argument('--noise_strength', default=config.noise_strength, type=float)
parser.add_argument('--self_attention', default=config.self_attention, type=bool_parser)
parser.add_argument('--encode_frequency', default=config.encode_frequency, type=bool_parser)
parser.add_argument('--position_encoding', default=config.position_encoding, type=bool_parser)
parser.add_argument('--transform', default=config.transform, type=bool_parser)
parser.add_argument('--input_channels', default=config.input_channels, type=int)

### Architecture params

# Anchors
parser.add_argument('--scales_factor_low', default=config.scales_factor_low, type=int)
parser.add_argument('--scales_factor_high', default=config.scales_factor_high, type=int)

# Anchor Target Layer
parser.add_argument('--rpn_neg_label', default=config.rpn_neg_label, type=float)
parser.add_argument('--rpn_pos_label', default=config.rpn_pos_label, type=float)
parser.add_argument('--rpn_batchsize', default=config.rpn_batchsize, type=int)
parser.add_argument('--rpn_fg_fraction', default=config.rpn_fg_fraction, type=float)

# Proposal Layer
parser.add_argument('--pre_nms_topN', default=config.pre_nms_topN, type=int)
parser.add_argument('--min_threshold', default=config.min_threshold, type=int)
parser.add_argument('--nms_thresh', default=config.nms_thresh, type=float)
parser.add_argument('--post_nms_topN', default=config.post_nms_topN, type=int)
parser.add_argument('--post_nms_topN_eval', default=config.post_nms_topN_eval, type=int)
parser.add_argument('--pre_nms_topN_eval', default=config.pre_nms_topN_eval, type=int)

# Proposal Target Layer
parser.add_argument('--rcnn_batch_size', default=config.rcnn_batch_size, type=int)
parser.add_argument('--rcnn_fg_prop', default=config.rcnn_fg_prop, type=float)
parser.add_argument('--fg_threshold', default=config.fg_threshold, type=float)
parser.add_argument('--bg_threshold_lo', default=config.bg_threshold_lo, type=float)
parser.add_argument('--bg_threshold_hi', default=config.bg_threshold_hi, type=float)

# ROI Pooling
parser.add_argument('--roi_pool_h', default=config.roi_pool_h, type=int)
parser.add_argument('--roi_pool_w', default=config.roi_pool_w, type=int)
parser.add_argument('--hidden_size', default=config.hidden_size, type=int)
parser.add_argument('--top_pyramid_roi_size', default=config.top_pyramid_roi_size, type=int)
parser.add_argument('--rcnn_attention', default=config.rcnn_attention, type=bool_parser)
parser.add_argument('--dropout', default=config.dropout, type=float)

# Training
parser.add_argument('--lambda_reg_rpn_loss', default=config.lambda_reg_rpn_loss, type=float)
parser.add_argument('--lambda_reg_rcnn_loss', default=config.lambda_reg_rcnn_loss, type=float)
parser.add_argument('--lambda_freq_regul', default=config.lambda_freq_regul, type=float)
parser.add_argument('--learning_rate', default=config.learning_rate, type=float)
parser.add_argument('--n_epochs', default=config.n_epochs, type=int)
parser.add_argument('--batch_size', default=config.batch_size, type=int)
parser.add_argument('--val_size', default=config.val_size, type=int)
parser.add_argument('--validation_prop', default=config.validation_prop, type=float)
parser.add_argument('--save_every', default=config.save_every, type=int)
parser.add_argument('--scheduler_gamma', default=config.scheduler_gamma, type=float)
parser.add_argument('--cv_idx', default=config.cv_idx, type=int)


## Arguments parsing and config parameters setting
args = parser.parse_args()

for attr, attr_value in args.__dict__.items():
    setattr(config, attr, attr_value)

## Looking for a previous checkpoint
save_dir = os.path.join(config.save_dir, config.model_name)
resume = False
if os.path.isdir(save_dir):
    resume = True

## Dataset instanciation
img_db_list = ['train_img_db_biophonia.hdf5', 'train_img_db_biophonia_2.hdf5', 'train_img_db_biophonia_noise.hdf5']
annot_list = ['train_annotations_biophonia.hdf5', 'train_annotations_biophonia_2.hdf5', 'train_annotations_biophonia_noise.hdf5']
img_db_list += [
    'extra_xc_img_db_fold_0.hdf5',
    'extra_xc_img_db_fold_1.hdf5',
    'extra_xc_img_db_fold_2.hdf5',
    'xc_img_db_fold_1.hdf5',
    'xc_img_db_fold_2.hdf5',
    'xc_img_db_fold_3.hdf5'
    ]
annot_list += [
    'extra_xc_annotations_fold_0.hdf5',
    'extra_xc_annotations_fold_1.hdf5', 
    'extra_xc_annotations_fold_2.hdf5', 
    'xc_annotations_fold_1.hdf5',
    'xc_annotations_fold_2.hdf5',
    'xc_annotations_fold_3.hdf5'    
    ]

img_db_path = [os.path.join(config.data_path, file) for file in img_db_list]
annotations_path = [os.path.join(config.data_path, file) for file in annot_list]
dataset = Img_dataset(img_db_path, annotations_path, normalize=config.normalize_input, zero_max=config.zero_max, noise_strength=config.noise_strength,
 transform=config.transform, fold=config.cv_idx)

## Train & validation sets
train_indices, val_indices = train_test_split(len(dataset), val_prop=config.validation_prop)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn)
validation_loader = DataLoader(dataset, batch_size=config.val_size, sampler=valid_sampler, collate_fn=collate_fn)

n_epochs = config.n_epochs

## Model instanciation and resuming
if config.backbone == 'vgg':
    conv_backbone = vgg16_bn(pretrained=config.pretrain_conv_net, pretrained_path=config.conv_net_path, input_channels=config.input_channels,
        fpn=config.fpn, fpn_p_channels=config.fpn_p_channels, fpn_o_channels=config.fpn_o_channels, self_attention=config.self_attention, 
        encode_frequency=config.encode_frequency, position_encoding=config.position_encoding).to(config.device)
else:
    conv_backbone = resnet50(pretrained=config.pretrain_conv_net, pretrained_path=config.conv_net_path, fpn=config.fpn, fpn_p_channels=config.fpn_p_channels,
     fpn_o_channels=config.fpn_o_channels, self_attention=config.self_attention, encode_frequency=config.encode_frequency, 
     position_encoding=config.position_encoding).to(config.device)
model = Faster_RCNN(config, conv_backbone).to(config.device)
epoch = 0

if resume:
    epoch = model.resume(save_dir)

## Training
gen_iterator = iter(train_loader)
validation_iterator = iter(validation_loader)
n_steps_per_epoch = len(train_loader)
save_epochs = []
while epoch < n_epochs:

    save_increment = 0
    
    for batch in iter(train_loader):
        
        ## Model update
        try:
            model.batch_update(batch)
        except IndexError:
            model.save(save_dir, epoch, save_increment, n_steps_per_epoch)
            epoch = n_epochs
            break

        # Validation
        if (save_increment + 1) % config.save_every == 0:
        
            valid_batch = next(validation_iterator, None)
            if valid_batch is None:
                validation_iterator = iter(validation_loader)
                valid_batch = validation_iterator.next()
            try:
                model(valid_batch)
            except IndexError:
                continue
                       
            # Average running losses and save
            model.update_learning_params()
            model.save(save_dir, epoch, save_increment, n_steps_per_epoch)

        save_increment += 1

    model.save(save_dir, epoch, save_increment, n_steps_per_epoch, new_file=epoch in save_epochs)
    epoch += 1
    model.step_scheduler()

# Save the configuration
args_serialize_path = os.path.join(save_dir, 'args')
with open(args_serialize_path, 'w') as f:
    json.dump(args.__dict__, f)