import os
import argparse
import json
from nets.nets_utils import *
from nets.effdet_layers import *
from pytorch_dataset.image_dataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def save(out_dir, model, optim, scheduler, train_loader, validation_loader, epoch, steps, best_val_cls_loss, label):

    save_dict = dict(
        checkpoints=model.state_dict(),
        optimizer=optim.state_dict(),
        scheduler=scheduler.state_dict(),
        loader=train_loader,
        val_loader=validation_loader,
        steps=steps,
        epoch=epoch,
        best_val_cls_loss=best_val_cls_loss
    )
    torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_' + label + '.pt'))

def resume(model, optim, scheduler, out_dir):
    
    save_dict = torch.load(os.path.join(out_dir, 'model_chkpt_last.pt'))
    model.load_state_dict(save_dict['checkpoints'])
    optim.load_state_dict(save_dict['optimizer'])
    scheduler.load_state_dict(save_dict['scheduler'])

    return save_dict['loader'], save_dict['val_loader'], save_dict['epoch'], save_dict['steps'], save_dict['best_val_cls_loss']

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
    if os.path.isfile(os.path.join(save_dir, 'model_chkpt_last.pt')):
        resume = True

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
    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    validation_loader = DataLoader(dataset, batch_size=4 * config.batch_size, sampler=valid_sampler, collate_fn=collate_fn)

    ## Model instanciation and resuming
    model = EffDet(config).to(config.device)
    optim = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, config.scheduler_milestones, config.scheduler_gamma)
    epoch, steps, best_val_cls_loss = 0, 0, 99
    if resume:
        train_loader, validation_loader, epoch, steps, best_val_cls_loss = resume(model, optim, scheduler, save_dir)

    ## Tensorboard
    writer = SummaryWriter(save_dir)

    ## Training

    while epoch < config.n_epochs:
        
        for batch in train_loader:
            
            ## Model update
            optim.zero_grad()
            cls_loss, reg_loss, neg_cls_loss = model.step(batch)
            total_loss = cls_loss + reg_loss + neg_cls_loss
            total_loss.backward()
            optim.step()
            writer.add_scalar(f'Training_Loss/cls_loss', cls_loss, global_step=steps)
            writer.add_scalar(f'Training_Loss/reg_loss', reg_loss, global_step=steps)
            writer.add_scalar(f'Training_Loss/neg_cls_loss', neg_cls_loss, global_step=steps)

            steps += 1

            # Validation
            if steps % 200 == 0:
                val_cls_loss, val_reg_loss, val_neg_cls_loss = 0, 0, 0
                for i, valid_batch in enumerate(validation_loader):
                    with torch.no_grad():
                        cls_loss, reg_loss, neg_cls_loss = model.step(valid_batch)
                    val_cls_loss += cls_loss
                    val_reg_loss += reg_loss
                    val_neg_cls_loss += neg_cls_loss
                val_cls_loss /= i
                val_reg_loss /= i
                val_neg_cls_loss /= i
                writer.add_scalar(f'Val_Loss/cls_loss', val_cls_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/reg_loss', val_reg_loss, global_step=steps)
                writer.add_scalar(f'Val_Loss/neg_cls_loss', val_neg_cls_loss, global_step=steps)

                if (epoch > 3) and (val_cls_loss < best_val_cls_loss):
                    best_val_cls_loss = val_cls_loss
                    save(save_dir, model, optim, scheduler, train_loader, validation_loader, epoch, steps, best_val_cls_loss, 'best')

        save(save_dir, model, optim, scheduler, train_loader, validation_loader, epoch, steps, best_val_cls_loss, 'last')
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

    # Focal loss
    parser.add_argument('--gamma', default=config.gamma, type=float)
    parser.add_argument('--alpha', default=config.alpha, type=float)

    ### Training

    parser.add_argument('--learning_rate', default=config.learning_rate, type=float)
    parser.add_argument('--n_epochs', default=config.n_epochs, type=int)
    parser.add_argument('--batch_size', default=config.batch_size, type=int)
    parser.add_argument('--validation_prop', default=config.validation_prop, type=float)
    parser.add_argument('--scheduler_gamma', default=config.scheduler_gamma, type=float)
    
    ## Arguments parsing
    args = parser.parse_args()

    ## Train
    train(args, config)