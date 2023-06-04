import os
import numpy as np
import torch
import torch.nn as nn
import pickle
from .faster_utils import *
from .layers import *



class Faster_RCNN(nn.Module):
    
    def __init__(self, config, conv_backbone):
        super(Faster_RCNN, self).__init__()
        
        self.config = config
        if self.config.classification:
            setattr(self.config, 'num_classes', 133) # remettre à 130
        
        # Classifier backbone
        self.conv_backbone = conv_backbone
        
        # Region Proposal Network
        if config.fpn_rpn:
            self.rpn = RegionProposalNetwork(config, in_channels=config.fpn_o_channels, out_channels=config.fpn_o_channels)
        elif config.backbone == 'resnet':
            self.rpn = RegionProposalNetwork(config, in_channels=2048)
        else:
            self.rpn = RegionProposalNetwork(config)
        
        # Anchor Target Layer
        self.anchor_target_layer = AnchorTargetLayer()
        
        # Proposal Layer
        self.prop_layer = ProposalLayer()
        
        # Proposal Target Layer
        self.proposal_target_layer = ProposalTargetLayer()
        
        # Fast-RCNN
        self.fast_rcnn = FastRCNN(config)
        
        # Optimizers
        betas = (0.5, 0.999)
        self.optim_params = list(self.conv_backbone.parameters()) + list(self.rpn.parameters()) \
            + list(self.fast_rcnn.parameters())
        self.optimizer = torch.optim.Adam(params=self.optim_params, betas=betas, lr=config.learning_rate)

        # Scheduler
        if config.scheduler_gamma < 1.0:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.scheduler_milestones, gamma=config.scheduler_gamma)


        ## Dictionary to track losses
        self.log_dict = {}
        loss_keys = ['rpn_classif', 'rpn_reg', 'rcnn_classif', 'rcnn_reg', 'mean_cls_score',
        'rcnn_in_pos_rois_per_bbox', 'rcnn_freq_loss']
        self.log_dict.update({
            'scores': {loss: [] for loss in loss_keys}
        })
        self.log_dict.update({
            'running_scores': {loss: [] for loss in loss_keys}
        })
        self.log_dict['valid_scores'] = []
        
        
    def forward(self, batch, evaluation=True, nms_thresh=0.3, min_score=0.5, inter_nms_thresh=0.3):
        
        if evaluation:
            img, gt_bbox, lengths, bird_ids, img_infos = batch
            img, gt_bbox = img.to(self.config.device), gt_bbox.to(self.config.device)
            
        else:
            img = batch.to(self.config.device)
        
        with torch.no_grad():
            conv_out = self.conv_backbone(img)
            if self.config.fpn:
                conv_out, last_conv_out = conv_out
                if self.config.fpn_rpn:
                    rpn_input = conv_out
                else:
                    rpn_input = last_conv_out
            else:
                rpn_input = conv_out

            cls_scores, bbox_reg = self.rpn(rpn_input)
            rois, cls_scores = self.prop_layer(cls_scores, bbox_reg, self.config, training=False)
            if len(rois) == 0:
                return
            output = self.fast_rcnn(rois, conv_out, training=False, inter_nms_thresh=inter_nms_thresh, intra_nms_thresh=nms_thresh, min_score=min_score)
        
        if evaluation:
            mAPs = 0
            indexes = np.cumsum([0] + lengths)
            for i, (i_0, i_f) in enumerate(zip(indexes[:-1], indexes[1:])):
                anchors = output[i]['1']['bbox_coord']
                bbox = gt_bbox[i_0:i_f]
                if len(anchors) == 0:
                    overlaps = torch.zeros(1, len(gt_bbox[i_0:i_f]))
                else:
                    overlaps = bbox_overlap(anchors, bbox)
                ovlp, assigned_box = overlaps.max(dim=1)

                # for each bbox, find the rank of the first output that fulfills a given minimum recall value
                max_rank = len(overlaps)
                ranks = []

                for j in range(overlaps.shape[1]):

                    j_idx = torch.nonzero(assigned_box == j)[:, 0]
                    ovlp_j = ovlp[j_idx]
                    ranks_j = []

                    for recall in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

                        above_idx = torch.nonzero(ovlp_j > recall)[:, 0]
                        if len(above_idx) == 0:
                            rank_j = len(ovlp_j)
                        else:
                            rank_j = above_idx[0].item()
                        ranks_j.append(rank_j)

                    ranks.append(ranks_j)

                mAP = (max_rank - np.array(ranks).sum(axis=0).mean())/max_rank
                mAPs += mAP
            
            mAPs = mAPs/(i + 1)
            print(f'Approx. mAP: {mAPs}')

            self.log_dict['valid_scores'].append(mAPs)

        else:
            return output
    
        
    def process_batch(self, batch):

        verbose = self.config.verbose
        
        img, gt_bbox, lengths, bird_ids, img_infos = batch
        img, gt_bbox = img.to(self.config.device), gt_bbox.to(self.config.device)
        
        conv_out = self.conv_backbone(img)

        if self.config.fpn:
            conv_out, last_conv_out = conv_out
            if self.config.fpn_rpn:
                rpn_input = conv_out
            else:
                rpn_input = last_conv_out
            # in this case the output from conv net backbone is a bottom-up list (the feature pyramid)
            height, width = conv_out[-1].shape[-2:]
        else:
            rpn_input = conv_out
            height, width = conv_out.shape[-2:]
        
        # RPN Targets
        with torch.no_grad():
            labels, reg_targets = self.anchor_target_layer(gt_bbox, lengths, self.config, height, width)
        
        # RPN + loss
        cls_scores, bbox_reg = self.rpn(rpn_input)

        ###
        batch_size, B, h_, w_ = cls_scores.shape
        K = h_ * w_
        A = int(B/2)
        mean_cls_score = cls_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, K * A, 2)[..., 1].mean().item()
        if verbose:
            print(f'Mean rpn classif score: {mean_cls_score:.4f}')
        self.log_dict['running_scores']['mean_cls_score'].append(mean_cls_score)
        ###

        rpn_class_loss, rpn_regression_loss = self.rpn.compute_losses(cls_scores, bbox_reg, labels, reg_targets)

        # Proposals + targets - no gradients are expected to flow past this layer
        with torch.no_grad():
            all_rois, cls_scores = self.prop_layer(cls_scores, bbox_reg, self.config)

            # if no proposal
            if len(all_rois) == 0:
                rcnn_class_loss = 0
                rcnn_regression_loss = 0
                return rpn_class_loss, rpn_regression_loss, rcnn_class_loss, rcnn_regression_loss

            rois, bbox_targets, labels, frequencies = self.proposal_target_layer(all_rois, gt_bbox, bird_ids, lengths, self.config)

            ###
            rcnn_in_pos_rois_per_bbox = labels.sum().item() / np.array(lengths).sum()
            self.log_dict['running_scores']['rcnn_in_pos_rois_per_bbox'].append(rcnn_in_pos_rois_per_bbox)
            if verbose:
                print(f'Positive ROIs per bbox as RCNN input: {rcnn_in_pos_rois_per_bbox:.4f}')
            ###
        
        rcnn_class_loss, rcnn_regression_loss, rcnn_freq_loss = self.fast_rcnn.compute_losses(rois, bbox_targets, labels, frequencies, conv_out)
        
        return rpn_class_loss, rpn_regression_loss, rcnn_class_loss, rcnn_regression_loss, rcnn_freq_loss
    
    
    def batch_update(self, batch):
        
        self.optimizer.zero_grad()
        
        rpn_class_loss, rpn_regression_loss, rcnn_class_loss, rcnn_regression_loss, rcnn_freq_loss = self.process_batch(batch)
        
        total_loss = rpn_class_loss + self.config.lambda_reg_rpn_loss * rpn_regression_loss +\
            rcnn_class_loss + self.config.lambda_reg_rcnn_loss * rcnn_regression_loss + self.config.lambda_freq_regul * rcnn_freq_loss
        
        total_loss.backward()
                
        self.optimizer.step()
        
        print(f'Losses: rpn cls : {rpn_class_loss}, rpn reg: {rpn_regression_loss}, \
        rcnn cls : {rcnn_class_loss}, rcnn reg: {rcnn_regression_loss}, rcnn freq reg: {rcnn_freq_loss}, total: {total_loss} \n')


        self.log_dict['running_scores']['rpn_classif'].append(rpn_class_loss.item())
        self.log_dict['running_scores']['rpn_reg'].append(rpn_regression_loss.item())
        self.log_dict['running_scores']['rcnn_classif'].append(rcnn_class_loss.item())
        self.log_dict['running_scores']['rcnn_reg'].append(rcnn_regression_loss.item())
        self.log_dict['running_scores']['rcnn_freq_loss'].append(rcnn_freq_loss.item())
        
    
    
    def save(self, directory, epoch, iteration, n_steps_per_epoch, new_file=False):

        idx = epoch * n_steps_per_epoch + iteration

        if not os.path.isdir(directory):
            os.mkdir(directory)

        save_dict = {
            'conv_backbone': self.conv_backbone.state_dict(),
            'rpn': self.rpn.state_dict(),
            'fast_rcnn': self.fast_rcnn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter': idx,
            'epoch': epoch
        }
        
        if new_file:
            torch.save(save_dict, os.path.join(directory, f'model_chkpt_{idx}.pt'))
        else:
            torch.save(save_dict, os.path.join(directory, f'model_chkpt.pt'))
        
        # Log file        
        log_file_path = os.path.join(directory, 'log_file')

        with open(log_file_path, 'wb') as f:
            pickle.dump(self.log_dict, f)


            
    def resume(self, checkpoint_dir, iteration=None):

        search = '.pt'
        if iteration is not None:
            search = f'model_chkpt_{iteration}' + search
        file = [f for f in os.listdir(checkpoint_dir) if search in f][0]

        checkpoint = torch.load(os.path.join(checkpoint_dir, file))

        # Networks
        self.conv_backbone.load_state_dict(checkpoint['conv_backbone'])
        self.rpn.load_state_dict(checkpoint['rpn'])
        self.fast_rcnn.load_state_dict(checkpoint['fast_rcnn'])

        # Optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        epoch = checkpoint['epoch']

        # Scheduler
        if self.config.scheduler_gamma < 1.0:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.scheduler_milestones, gamma=self.config.scheduler_gamma,
            last_epoch=epoch)
            
        # Log file
        log_file_path = os.path.join(checkpoint_dir, 'log_file')
        with open(log_file_path, 'rb') as f:
            self.log_dict = pickle.load(f)
            
        return epoch

    
    def update_learning_params(self):            
        
        loss_keys = ['rpn_classif', 'rpn_reg', 'rcnn_classif', 'rcnn_reg', 'mean_cls_score',
        'rcnn_in_pos_rois_per_bbox', 'rcnn_freq_loss']
        for key in loss_keys:
            if len(self.log_dict['running_scores'][key]) == 0:
                continue
            self.log_dict['scores'][key].append(np.array(self.log_dict['running_scores'][key]).mean())
            self.log_dict['running_scores'][key] = []


    def step_scheduler(self):

        if hasattr(self, 'scheduler'):
            self.scheduler.step()
