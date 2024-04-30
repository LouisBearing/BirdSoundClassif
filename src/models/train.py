# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
# import datetime
import json
# import random
# import time
# from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import nets.util.misc as utils
# from engine import evaluate, train_one_epoch
from nets import build_model

from nbm_datasets.image_dataset import *
from nets.util.nets_utils import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=383, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                     help="Train segmentation head if the flag is provided")

    # # Loss
    # parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
    #                     help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    ###
    parser.add_argument('--model_name', default='new_model', type=str)
    parser.add_argument('--data_path', default='dataset', type=str)
    parser.add_argument('--save_dir', default='models_detr', type=str)
    parser.add_argument('--pretrained_dir', default='pretr_weights', type=str)
    parser.add_argument('--max_steps', default=5e5, type=float)
    parser.add_argument('--first_neg_step', default=1e5, type=float)
    parser.add_argument('--neg_step_freq', default=10, type=int)
    parser.add_argument('--save_step', default=None, type=float)

    parser.add_argument('--num_classes', default=144, type=int)

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


def resume(out_dir, model, optim, scheduler, lr_drop):
    
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

    return loss_dict['loss_ce'].item(), loss_dict['loss_bbox'].item(), loss_dict['loss_giou'].item(), \
        loss_dict['loss_neg_ce'].item(), loss_dict['cardinality_error'].item()


def step(model, criterion, batch, device, negative_sample):

    img, neg_img, bb_coord, bird_ids, lengths = batch
    img, neg_img, bb_coord, bird_ids = img.to(device), neg_img.to(device), bb_coord.to(device), bird_ids.to(device)

    ## Map to detr format
    bb_coord_rel = coord_to_rel(bb_coord)
    targets = []
    indexes = np.cumsum([0] + lengths)
    for b_idx, (i_0, i_f) in enumerate(zip(indexes[:-1], indexes[1:])):
        targets.append({
            'labels': bird_ids[i_0:i_f].to(torch.int64),
            'boxes': bb_coord_rel[i_0:i_f]
        })

    if negative_sample:
        outputs = model(neg_img[:, None])
    else:
        outputs = model(img[:, None])
    loss_dict = criterion(outputs, targets, negative_sample)

    return loss_dict


def main(args):

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

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

    ## Dataset instanciation
    dataset = Img_dataset(args.data_path, transform=True)

    if resume:
        model, optimizer, lr_scheduler, train_indices, val_indices, epoch, steps, best_val_cls_loss = resume(save_dir, model, optimizer, lr_scheduler, args.lr_drop)
    else:
        train_indices, val_indices = train_test_split(len(dataset), val_prop=args.validation_prop)
        epoch, steps, best_val_cls_loss = 0, 0, 99

    ## Train & validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=args.num_workers)
    validation_loader = DataLoader(dataset, batch_size=4 * args.batch_size, sampler=valid_sampler, collate_fn=collate_fn,
     num_workers=args.num_workers, drop_last=True)

    print("Start training")

    ## Tensorboard
    writer = SummaryWriter(save_dir)

    ## Training
    train_losses = {
        'cls_loss': 0,
        'reg_loss': 0,
        'giou_loss': 0,
        'neg_cls_loss': 0,
        'cardinality_error': 0
    }

    save_steps = [150e3, 200e3, 250e3] # args.save_step
    while steps < args.max_steps:

        for batch in train_loader:
            cls_loss, reg_loss, giou_loss, neg_cls_loss, cardinality_error = train_one_step(
                model, criterion, optimizer, batch, args.clip_max_norm, device, 
                negative_sample=(steps % args.neg_step_freq == 0) and (steps > args.first_neg_step))
            train_losses['cls_loss'] += cls_loss
            train_losses['reg_loss'] += reg_loss
            train_losses['giou_loss'] += giou_loss
            train_losses['neg_cls_loss'] += neg_cls_loss
            train_losses['cardinality_error'] += cardinality_error
            if steps % 50 == 0:
                for key in train_losses.keys():
                    writer.add_scalar(f'Training_Loss/{key}', train_losses[key] / 50, global_step=steps)
                    train_losses[key] = 0

            if steps in save_steps:
                save(save_dir, model, epoch, steps, best_val_cls_loss, str(steps), optimizer, lr_scheduler, train_indices, val_indices)

            steps += 1

            # Validation
            if steps % 200 == 0:
                model.eval(), criterion.eval()
                val_cls_loss, val_reg_loss, val_giou_loss, val_card_error = 0, 0, 0, 0
                for i, valid_batch in enumerate(validation_loader):
                    with torch.no_grad():
                        loss_dict = step(model, criterion, valid_batch, device, negative_sample=False)
                    val_cls_loss += loss_dict['loss_ce'].item()
                    val_reg_loss += loss_dict['loss_bbox'].item()
                    val_giou_loss += loss_dict['loss_giou'].item()
                    val_card_error += loss_dict['cardinality_error'].item()
                val_cls_loss /= i
                val_reg_loss /= i
                val_giou_loss /= i
                val_card_error /= i
                with torch.no_grad():
                    loss_dict = step(model, criterion, valid_batch, device, negative_sample=True)
                val_neg_cls_loss = loss_dict['loss_neg_ce'].item()
                writer.add_scalar(f'Val_Loss/cls_loss', val_cls_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/reg_loss', val_reg_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/giou_loss', val_giou_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/neg_cls_loss', val_neg_cls_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/cardinality_error', val_card_error, global_step=steps)
            
                if (epoch > 3) and (val_cls_loss < best_val_cls_loss):
                    best_val_cls_loss = val_cls_loss
                    save(save_dir, model, epoch, steps, best_val_cls_loss, 'best')

                model.train(), criterion.train()

        if (epoch > 20) and (epoch % 5 == 0):
            save(save_dir, model, epoch, steps, best_val_cls_loss, 'last', optimizer, lr_scheduler, train_indices, val_indices)
        lr_scheduler.step()
        writer.add_scalar(f'Lr', lr_scheduler.get_last_lr()[0], global_step=steps)

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
