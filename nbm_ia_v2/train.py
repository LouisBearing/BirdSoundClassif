import os
import argparse
import json
from nets.nets_utils import *
from nets.effdet_layers import *
from pytorch_dataset.image_dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def save(out_dir, model, epoch, steps, best_val_cls_loss, label):

    save_dict = dict(
        checkpoints=model.state_dict(),
        steps=steps,
        epoch=epoch,
        best_val_cls_loss=best_val_cls_loss
    )
    torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_' + label + '.pt'))


# def save(out_dir, model, optim, scheduler, train_loader, validation_loader, epoch, steps, best_val_cls_loss, label):

#     save_dict = dict(
#         checkpoints=model.state_dict(),
#         optimizer=optim.state_dict(),
#         scheduler=scheduler.state_dict(),
#         loader=train_loader,
#         val_loader=validation_loader,
#         steps=steps,
#         epoch=epoch,
#         best_val_cls_loss=best_val_cls_loss
#     )
#     torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_' + label + '.pt'))

# def resume(model, optim, scheduler, out_dir):
    
#     save_dict = torch.load(os.path.join(out_dir, 'model_chkpt_last.pt'))
#     model.load_state_dict(save_dict['checkpoints'])
#     optim.load_state_dict(save_dict['optimizer'])
#     scheduler.load_state_dict(save_dict['scheduler'])

#     return save_dict['loader'], save_dict['val_loader'], save_dict['epoch'], save_dict['steps'], save_dict['best_val_cls_loss']

def step_scheduler(scheduler):
    scheduler.step()
    return scheduler.get_last_lr()[0]

def train(args, config):

    for attr, attr_value in args.__dict__.items():
        setattr(config, attr, attr_value)

    ## Looking for a previous checkpoint
    save_dir = os.path.join(config.save_dir, config.model_name)
    os.makedirs(save_dir, exist_ok=True)
    resume = False
    # if os.path.isfile(os.path.join(save_dir, 'model_chkpt_last.pt')):
    #     resume = True

    ## Save the configuration
    args_serialize_path = os.path.join(save_dir, 'args')
    with open(args_serialize_path, 'w') as f:
        json.dump(args.__dict__, f)

    ## Dataset instanciation
    dataset = Img_dataset(config.data_path, transform=True)

    ## Train & validation sets
    train_indices, val_indices = train_test_split(len(dataset), val_prop=config.validation_prop)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
    validation_loader = DataLoader(dataset, batch_size=4 * config.batch_size, sampler=valid_sampler, collate_fn=collate_fn, num_workers=4)

    ## Model instanciation and resuming
    model = EffDet(config).to(config.device)
    # parameters = [
    #     {'params': list(model.backbone.parameters()) + list(model.fpn.parameters()) + list(model.branches.class_branch.parameters()) \
    #         + list(model.branches.box_branch.parameters())}]
    # if config.bias_p > 0:
    #     parameters.append([
    #         {'params': model.branches.out_bias.parameters(), 'lr': config.learning_rate_bias_out}
    #     ])
    # if config.pre_fpn_attn:
    #     parameters[0]['params'].extend(list(model.attn_modules.parameters()))
    parameters = model.parameters()
    if config.optim == 'sgd':
        optim = torch.optim.SGD(parameters, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optim = torch.optim.Adam(parameters, betas=(config.momentum, 0.99), lr=config.learning_rate)
    if config.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, config.scheduler_milestones, config.scheduler_gamma)
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, int(config.cosine_scheduler_t0), config.cosine_scheduler_tmult,
            eta_min=1e-5)
    epoch, steps, best_val_cls_loss = 0, 0, 99
    # if resume:
    #     train_loader, validation_loader, epoch, steps, best_val_cls_loss = resume(model, optim, scheduler, save_dir)


    ## Tensorboard
    writer = SummaryWriter(save_dir)

    ## Training
    train_losses = {
        'cls_loss': 0,
        'reg_loss': 0,
        'neg_cls_loss': 0
    }
    while epoch < config.n_epochs:
        
        for batch in train_loader:

            # alpha = scheduler_fn_alpha(steps, a_bar_max=1 - config.alpha, max_step=25000)
            # alpha = 0.9999
            alpha = 0.5
            
            ## Model update
            optim.zero_grad()
            # neg_step=epoch >= config.first_epoch_neg_loss,
            cls_loss, reg_loss, neg_cls_loss = model.step(batch, neg_step=steps % 20 == 0, alpha=alpha)
            total_loss = cls_loss + reg_loss + neg_cls_loss
            total_loss.backward()
            optim.step()
            train_losses['cls_loss'] += cls_loss
            train_losses['reg_loss'] += reg_loss
            train_losses['neg_cls_loss'] += neg_cls_loss
            if steps % 50 == 0:
                for key in train_losses.keys():
                    writer.add_scalar(f'Training_Loss/{key}', train_losses[key] / 50, global_step=steps)
                    train_losses[key] = 0

            steps += 1

            # Validation
            if steps % 200 == 0:
                val_cls_loss, val_reg_loss, val_neg_cls_loss = 0, 0, 0
                for i, valid_batch in enumerate(validation_loader):
                    with torch.no_grad():
                        cls_loss, reg_loss, _ = model.step(valid_batch, neg_step=False,
                            alpha=alpha, gamma=1.5)
                    val_cls_loss += cls_loss
                    val_reg_loss += reg_loss
                val_cls_loss /= i
                val_reg_loss /= i
                with torch.no_grad():
                    _, _, val_neg_cls_loss = model.step(valid_batch, neg_step=True,
                                alpha=alpha, gamma=1.5)
                writer.add_scalar(f'Val_Loss/cls_loss', val_cls_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/reg_loss', val_reg_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/neg_cls_loss', val_neg_cls_loss, global_step=steps)

                if (epoch > 3) and (val_cls_loss < best_val_cls_loss):
                    best_val_cls_loss = val_cls_loss
                    save(save_dir, model, epoch, steps, best_val_cls_loss, 'best')

        save(save_dir, model, epoch, steps, best_val_cls_loss, 'last')
        epoch += 1
        lr = step_scheduler(scheduler)
        writer.add_scalar(f'Lr', lr, global_step=steps)


if __name__ == "__main__":

    config = Config()
    parser = argparse.ArgumentParser()

    ### General params

    parser.add_argument('--model_name', default='new_model', type=str)
    parser.add_argument('--data_path', default='dataset', type=str)
    parser.add_argument('--save_dir', default=config.save_dir, type=str)
    parser.add_argument('--bckb', default=config.bckb, type=str)

    ### Architecture params

    # Anchors
    parser.add_argument('--base_size', default=config.base_size, type=int)

    # Detector
    parser.add_argument('--n_classes', default=config.n_classes, type=int)
    parser.add_argument('--dropout', default=config.dropout, type=float)
    parser.add_argument('--n_layers_bifpn', default=config.n_layers_bifpn, type=int)
    parser.add_argument('--out_channels', default=config.out_channels, type=int)
    parser.add_argument('--n_layers_branches', default=config.n_layers_branches, type=int)
    parser.add_argument('--min_score', default=config.min_score, type=float)
    parser.add_argument('--inter_nms_thresh', default=config.inter_nms_thresh, type=float)
    parser.add_argument('--intra_nms_thresh', default=config.intra_nms_thresh, type=float)
    parser.add_argument('--pre_fpn_attn', default=config.pre_fpn_attn, type=bool_parser)
    parser.add_argument('--c1d_branches', default=config.c1d_branches, type=bool_parser)
    parser.add_argument('--c1d_div_fact', default=config.c1d_div_fact, type=int)
    parser.add_argument('--n_samples_ce', default=config.n_samples_ce, type=int)
    parser.add_argument('--expansion_fact_fpn', default=config.expansion_fact_fpn, type=int)

    # Focal loss
    parser.add_argument('--gamma', default=config.gamma, type=float)
    parser.add_argument('--alpha', default=config.alpha, type=float)
    parser.add_argument('--bias_p', default=config.bias_p, type=float)

    ### Training
    parser.add_argument('--learning_rate', default=config.learning_rate, type=float)
    parser.add_argument('--learning_rate_bias_out', default=config.learning_rate_bias_out, type=float)
    parser.add_argument('--n_epochs', default=config.n_epochs, type=int)
    parser.add_argument('--batch_size', default=config.batch_size, type=int)
    parser.add_argument('--validation_prop', default=config.validation_prop, type=float)
    parser.add_argument('--scheduler_gamma', default=config.scheduler_gamma, type=float)
    parser.add_argument('--first_epoch_neg_loss', default=config.first_epoch_neg_loss, type=int)
    parser.add_argument('--optim', default=config.optim, type=str)
    parser.add_argument('--momentum', default=config.momentum, type=float)
    parser.add_argument('--weight_decay', default=config.weight_decay, type=float)
    parser.add_argument('--scheduler', default=config.scheduler, type=str)
    parser.add_argument('--cosine_scheduler_t0', default=config.cosine_scheduler_t0, type=int)
    parser.add_argument('--cosine_scheduler_tmult', default=config.cosine_scheduler_tmult, type=int)
    
    ## Arguments parsing
    args = parser.parse_args()

    ## Train
    train(args, config)