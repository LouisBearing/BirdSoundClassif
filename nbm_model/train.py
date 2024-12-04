# Copyright (c) NBM. All Rights Reserved
import argparse
import os
import json
import random
import torch
import glob
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from nets.util.nets_utils import collate_fn, setattr_others, train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from nets import build_model
from nbm_datasets.image_dataset import Img_dataset
from nets.util.nets_utils import (collate_fn, setattr_others, train_test_split, 
                                  format_txt_annots, compute_AP_scores)
from run_detection import run_detection



def get_args_parser():
    
    # General params
    parser = argparse.ArgumentParser('Set detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=383, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--model_name', default='new_model', type=str)
    parser.add_argument('--data_path', default='dataset', type=str)
    parser.add_argument('--save_dir', default='models', type=str)
    parser.add_argument('--max_steps', default=5e5, type=float)
    parser.add_argument('--first_neg_step', default=0, type=float)
    parser.add_argument('--neg_step_freq', default=10, type=int)
    parser.add_argument('--save_step', default=None, type=float)
    parser.add_argument('--img_width', default=1024, type=int)
    parser.add_argument('--img_height', default=375, type=int)
    parser.add_argument('--inpt_channels', default=1, type=int,
                        help='Number of input channels of the original images')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    '''Possible values: 'vgg16_bn', 'resnet50', 'resnet101', 'resnext101_32x8d' or any other resnet, 'efficientnet_bN' with N <= 4,
        efficientnet_v2_m, efficientnet_v2_l
    '''
    
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--add_posenc', action='store_true', default=False,
                        help="If true, add the positional encoding to the feature maps between backbone and SA layers")
    parser.add_argument('--one_dim_posenc', default=True, action='store_false')
    parser.add_argument('--norm_layer_backbone', default='frozen_batchnorm', type=str)

    # * Loss coefficients
    parser.add_argument('--fs_cls_loss_coef', default=1, type=float)
    parser.add_argument('--fs_neg_cls_loss_coef', default=1, type=float)
    parser.add_argument('--fs_reg_loss_coef', default=1, type=float)
    parser.add_argument('--sec_cls_loss_coef', default=1, type=float)
    parser.add_argument('--sec_neg_cls_loss_coef', default=1, type=float)
    parser.add_argument('--sec_reg_loss_coef', default=1, type=float)
    parser.add_argument('--focal_loss', action='store_true', default=False,
                    help="Whether to apply focal loss or CE for second stage classification")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)    

    # Anchors & FRCNN
    parser.add_argument('--n_ratios', default=3, type=int)
    parser.add_argument('--anchor_stride', default=16, type=int)
    parser.add_argument('--base_size', default=16, type=int)

    parser.add_argument('--rpn_neg_label', default=0.3, type=float,
                        help='Max IoU threshold with a GT box for an anchor to be considered negative')
    parser.add_argument('--rpn_pos_label', default=0.7, type=float,
                        help='Min IoU threshold with a GT box for an anchor to be considered positive')
    parser.add_argument('--rpn_batchsize', default=16, type=int,
                        help='Number of RoIs output by the AnchorTargetLayer module to train the RPN')
    parser.add_argument('--rpn_fg_fraction', default=0.5, type=float,
                        help='Fraction of positive RoIs for RPN training')

    parser.add_argument('--rcnn_batch_size', default=16, type=int,
                        help='Number of box output by the ProposalTargetLayer module to train the RCNN')
    parser.add_argument('--rcnn_fg_prop', default=0.4, type=float,
                        help='Fraction of positive boxes for 2n stage training') # 0.25 in the original paper
    parser.add_argument('--fg_threshold', default=0.5, type=float,
                        help='Min IoU threshold with a GT box for a proposal box to be considered foreground')
    parser.add_argument('--bg_threshold_lo', default=0.1, type=float,
                        help='Min IoU threshold with a GT box for a proposal box to be considered background (ignored otherwise)')
    parser.add_argument('--bg_threshold_hi', default=0.5, type=float,
                        help='Max IoU threshold with a GT box for a proposal box to be considered background')
    parser.add_argument('--depth_rcnn', default=3, type=int,
                        help='Number of layers in the RCNN network, if rcnn_pe option is chosen')
    
    parser.add_argument('--pre_nms_topN', default=3000, type=int,
                        help='Filtered number of RoIs before applying nms in proposal layer') ## Bottleneck arg
    parser.add_argument('--min_threshold', default=5, type=int,
                        help='Min number of box side in pixels under which RoI proposals are discarded')
    parser.add_argument('--nms_thresh', default=0.7, type=float,
                        help='Min IoU threshold above which nms discards one of the bboxes')
    parser.add_argument('--post_nms_topN', default=1000, type=int,
                        help='Number of selected RoIs after nms')
    parser.add_argument('--post_nms_topN_eval', default=50, type=int,
                        help='Number of selected RoIs after nms in eval mode')
    parser.add_argument('--pre_nms_topN_eval', default=500, type=int,
                        help='Number of RoIs inputs for nms in eval mode')
    parser.add_argument('--roi_pool_h', default=2, type=int,
                        help='Height of RoI pooling layer output in pix')
    parser.add_argument('--roi_pool_w', default=2, type=int,
                        help='Width of RoI pooling layer output in pix')
    parser.add_argument('--hidden_size_rcnn', default=512, type=int,
                        help='Intermediate dimension of RCNN linear layers')
    parser.add_argument('--dropout', default=0, type=float,
                        help='Dropout probability')
    parser.add_argument('--proposal_number', default=50, type=int,
                        help='Number of proposals per class after last nms')

    # FPN
    parser.add_argument('--fpn', default='fpn', type=str,
                        help='Which FPN layer: bifpn or fpn')
    parser.add_argument('--n_bifpn_layers', default=5, type=int)
    parser.add_argument('--fpn_p_chan', default=384, type=int)
    parser.add_argument('--out_fpn_chan', default=256, type=int)
    # parser.add_argument('--layered_rcnn', action='store_true', default=False,
    #                     help="If true, use different networks for distinct scales in the RCNN")
    # parser.add_argument('--rcnn_pe', action='store_true', default=False,
    #                     help="If true, encode frequency and RoI width in the RCNN")
    parser.add_argument('--fpn_first', action='store_true', default=False,
                        help="If true, applies fpn followed by attn, otherwise, if sandwich attn is also false, \
                        does the opposite")
    parser.add_argument('--sandwich_attn', action='store_true', default=False,
                        help="If true, applies attn, then fpn and then another attn layer")

    # Tf_RCNN
    parser.add_argument('--tf_rcnn', action='store_true', default=False,
                    help="If true, replace RCNN by a Transformer")
    parser.add_argument('--tf_pe_qk', action='store_true', default=False,
                    help="If true, add posenc at each tf layer to Q & K (but not V)")
    parser.add_argument('--tf_model_dim', default=512, type=int,
                    help='Inner dim of the TF RCNN')
    parser.add_argument('--tf_nhead', default=8, type=int,
                    help='Number of heads of the TF RCNN')
    parser.add_argument('--tf_num_encoder_layers', default=6, type=int,
                    help='Number of TF RCNN encoder layers')
    parser.add_argument('--tf_dim_feedforward', default=1024, type=int,
                    help='Inner dim of TF RCNN FF layers')

    # Self Attention
    # parser.add_argument('--no_attn', action='store_true', default=False,
    #                     help="If true, no SA layer is used")
    parser.add_argument('--pyramid_top_n_attn', default=2, type=int)
    parser.add_argument('--num_classes', default=150, type=int)
    parser.add_argument('--validation_prop', default=0.03, type=float)
    ###

    # # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def save(out_dir, model, epoch, steps, best_val_cls_loss, label, optim=None, scheduler=None, train_indices=None, val_indices=None):

    save_dict = dict(
        checkpoints=model.state_dict(),
        steps=steps,
        epoch=epoch,
        best_val_cls_loss=best_val_cls_loss
    )
    if optim is not None:
        save_dict.update({'optim': optim.state_dict()})
    if scheduler is not None:
        save_dict.update({'scheduler': scheduler.state_dict()})
    if train_indices is not None:
        save_dict.update({'train_indices': train_indices})
    if val_indices is not None:
        save_dict.update({'val_indices': val_indices})
    torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_' + label + '.pt'))


def resume_training(out_dir, model, optim, scheduler, lr_drop):
    
    save_dict = torch.load(os.path.join(out_dir, 'model_chkpt_last.pt'))

    model.load_state_dict(save_dict['checkpoints'])
    model.train()
    optim.load_state_dict(save_dict['optim'])
    scheduler_state_dict = save_dict['scheduler']
    scheduler_state_dict['step_size'] = lr_drop
    scheduler.load_state_dict(scheduler_state_dict)

    return model, optim, scheduler, save_dict['train_indices'], save_dict['val_indices'], save_dict['epoch'], save_dict['steps'], \
        save_dict['best_val_cls_loss']


def train_one_step(model, criterion, optimizer, batch, max_norm, device, negative_sample):
    
    loss_dict = step(model, criterion, batch, device, negative_sample)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    optimizer.zero_grad()
    losses.backward()
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    return loss_dict


def step(model, criterion, batch, device, negative_sample):

    img, neg_img, bb_coord, bird_ids, lengths = batch
    img, neg_img, bb_coord, bird_ids = img.to(device), neg_img.to(device), bb_coord.to(device), bird_ids.to(device)

    loss = {}
    if not negative_sample:
        inpt = img[:, None]
    else:
        inpt = neg_img[:, None]
        
    # First stage
    out_first_stage = model.forward_first_stage(inpt)
    first_stage_loss = criterion.first_stage_loss(out_first_stage['rpn_cls_scores'], out_first_stage['rpn_bbox_reg'], 
                                                  bb_coord, lengths, negative_sample)
    loss.update(first_stage_loss)
    # In some cases the RPN fails to generate enough proposals, in these cases there is no second stage loss
    if len(out_first_stage['rois']) == 0:
        return loss

    # Second stage
    if not negative_sample:
        proposal_tgt_out = criterion.generate_all_rois(out_first_stage['rois'], bb_coord, bird_ids, lengths)
        if proposal_tgt_out['rois'] is None:
            return loss
    else:
        proposal_tgt_out = {'rois': out_first_stage['rois'], 'bbox_targets': None, 'labels': None}
    out_second_stage = model.forward_second_stage(out_first_stage['fpn_out'], proposal_tgt_out['rois'], training=True)
    second_stage_loss = criterion.second_stage_loss(out_second_stage['bbox_reg'], out_second_stage['bbox_classes'],
                                                    proposal_tgt_out['bbox_targets'], proposal_tgt_out['labels'],
                                                    negative_sample)
    loss.update(second_stage_loss)

    # Cardinality error
    if not negative_sample:
        loss.update(criterion.loss_cardinality(out_second_stage['bbox_classes'], proposal_tgt_out['labels']))

    return loss


def seed_everything(seed: int) -> None: # From PytorchGeometrics
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    device = torch.device(args.device)
    seed_everything(args.seed)

    ## Looking for a previous checkpoint
    save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    resume = False
    if os.path.isfile(os.path.join(save_dir, 'model_chkpt_last.pt')):
        resume = True

    ## Save the configuration
    args_serialize_path = os.path.join(save_dir, 'args')
    with open(args_serialize_path, 'w') as f:
        json.dump(args.__dict__, f)

    setattr_others(args)

    model, criterion = build_model(args)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    ## Dataset instanciation
    dataset = Img_dataset(args.data_path, transform=True)

    if resume:
        model, optimizer, lr_scheduler, train_indices, val_indices, epoch, steps, best_val_cls_loss = resume_training(save_dir, model, optimizer, lr_scheduler, args.lr_drop)
        print('Resuming training~~~~')
    else:
        train_indices, val_indices = train_test_split(len(dataset), val_prop=args.validation_prop)
        epoch, steps, best_val_cls_loss = 0, 0, 99

    ## Train & validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    if len(val_indices) > 0:
        valid_sampler = SubsetRandomSampler(val_indices)
        validation_loader = DataLoader(dataset, batch_size=2 * args.batch_size, sampler=valid_sampler, collate_fn=collate_fn,
        num_workers=args.num_workers, drop_last=True)

    print("Start training")

    ## Tensorboard
    writer = SummaryWriter(save_dir)

    ## Training
    loss_keys = ['first_class_loss', 'first_regression_loss', 'sec_class_loss', 'sec_regression_loss', 
                 'first_neg_class_loss', 'sec_neg_class_loss', 'cardinality_error']
    train_losses = {
        l: 0 for l in loss_keys
    }

    save_steps = [180e3, 190e3, 200e3] # args.save_step
    while steps < args.max_steps:

        for batch in train_loader:
            losses = train_one_step(
                model, criterion, optimizer, batch, args.clip_max_norm, device, 
                negative_sample=(steps % args.neg_step_freq == 0) and (steps > args.first_neg_step))
            for l in losses.keys():
                train_losses[l] += losses[l].item() if l != 'cardinality_error' else losses[l]
            if steps % 50 == 0:
                for l in loss_keys:
                    freq = 50 / args.neg_step_freq if 'neg' in l else 50
                    writer.add_scalar(f'Training_Loss/{l}', train_losses[l] / freq, global_step=steps)
                    train_losses[l] = 0

            if steps in save_steps:
                save(save_dir, model, epoch, steps, best_val_cls_loss, str(steps), optimizer, lr_scheduler, train_indices, val_indices)

            steps += 1
            if steps % 1000 == 0:
                lr_scheduler.step()
                writer.add_scalar(f'Lr', lr_scheduler.get_last_lr()[0], global_step=steps)

            # Validation
            if steps % 500 == 0:
                model.eval(), criterion.eval()
                val_losses = {
                    l: 0 for l in loss_keys
                }
                if args.validation_prop > 0:
                    # Positive samples
                    for i, valid_batch in enumerate(validation_loader):
                        with torch.no_grad():
                            loss_dict = step(model, criterion, valid_batch, device, negative_sample=False)
                        for l in loss_dict.keys():
                            val_losses[l] += loss_dict[l].item() if l != 'cardinality_error' else loss_dict[l]
                    for l in loss_keys:
                        val_losses[l] /= i
                    # One negative batch
                    with torch.no_grad():
                        loss_dict = step(model, criterion, valid_batch, device, negative_sample=True)
                    for l in loss_dict.keys():
                        val_losses[l] += loss_dict[l].item()
                    # Write losses
                    for l in loss_keys:
                        writer.add_scalar(f'Val_Loss/{l}', val_losses[l], global_step=steps)
                
                    if (steps / 1000 > args.lr_drop) and (val_losses['sec_class_loss'] < best_val_cls_loss):
                        best_val_cls_loss = val_losses['sec_class_loss']
                        save(save_dir, model, epoch, steps, best_val_cls_loss, 'best')

                # Test
                outputs = []
                for i, wav_path in enumerate(glob.glob(os.path.join(args.data_path, 'test_files', 'XC_annots') + '/*.wav')):
                    #### Exec model
                    output = run_detection(model, args, wav_path, min_score=0.02, bird_dicts_path='bird_dict.json')
                    outputs.append((output, format_txt_annots(wav_path.replace('.wav', '.txt'))))
                test_metrics = compute_AP_scores(outputs)
                for l in test_metrics.keys():
                    writer.add_scalar(f'Test_metrics/{l}', test_metrics[l], global_step=steps)

                model.train(), criterion.train()

        if (epoch > 0) and (epoch % 10 == 0):
            save(save_dir, model, epoch, steps, best_val_cls_loss, 'last', optimizer, lr_scheduler, train_indices, val_indices)

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NbmModel training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
