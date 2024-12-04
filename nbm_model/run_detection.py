import numpy as np
import os
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from nbm_model.nets.util.nets_utils import setattr_others
from nbm_model.nbm_datasets.prepare_dataset import File_Processor
from nbm_model.nets.backbone import build_backbone
from nbm_model.nets.fpn import build_fpn
from nbm_model.nets.self_attention import build_sa_layers
from nbm_model.nets.head import build_head
from nbm_model.nets.nbm_model import NbmModel, initialize_model
from nbm_model.nets.util.nets_utils import nms
from nbm_model.nets.util.nets_utils import rel_to_coord


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def run_detection(model, config, wav_path, bird_dicts_path, min_score=0.5, bs=10, visualise_outputs=False, show_sp_name=True):
    '''
    Params:
    ------
    model
    config
    wav_path
    min_score
    bs (int): batch size, how many samples processed at one
    return_spectrogram (bool)
    '''

    fp = File_Processor(wav_path)
    img_db, _ = fp.process_file()
    
    batch = []
    outputs = []
    spectrogram = []

    n_img = len(img_db)
    b_idx = 0
    for i in tqdm(range(n_img)):
        img = img_db[i]
        batch.append(img)
        if (len(batch) == bs) or (i == n_img - 1):
            batch = torch.Tensor(np.stack(batch)).to(device)
            with torch.no_grad():
                o = model(batch[:, None], min_score=min_score)
            # batch_out = postpro_detr(o, config, min_score=min_score) --> In two stage-models, no need for this
            batch_out = o
            outputs.append(batch_out)
            
            if visualise_outputs:
                for sample_id, sample in enumerate(batch_out):
                    boxes = [sample[str(b_id)]['bbox_coord'] for b_id in np.arange(1, len(sample)) if len(sample[str(b_id)]['bbox_coord'] > 0)]
                    if len(boxes) > 0:
                        idx = b_idx * bs + sample_id
                        spectrogram.append((idx, batch[sample_id]))  
            batch = []
            b_idx += 1

    ### Load bird id dictionary
    with open(bird_dicts_path, 'r') as f:
        birds_dict = json.load(f)
    birds_dict.update({'Non bird sound': 0})
    reverse_dict = {id: bird_name for bird_name, id in birds_dict.items()}

    ### Merge results into a file output
    class_bbox = merge_images(fp, outputs, config.num_classes)
    output = {reverse_dict[idx]: {key: value.cpu().numpy().tolist() for key, value in class_bbox[str(idx)].items()} for idx in range(1, len(class_bbox) + 1) if len(class_bbox[str(idx)]['bbox_coord']) > 0}
    
    ### Results vizualisation
    if visualise_outputs:
        visualise_model_out(output, fp, spectrogram, reverse_dict, show_sp_name)
        return output
    else:
        return output


def load_model(mod_p):

    # Dummy class to load model's arguments
    class Args:
        def __init__(self, **kwargs):
            for (k, v) in kwargs.items():
                setattr(self, k, v) 

    args_path = os.path.join(mod_p, 'args')
    with open(args_path, 'rb') as f:
        args = json.load(f)
    args = Args(**args)
    setattr_others(args)

    backbone = build_backbone(args)
    if args.fpn_first:
        attn_channels = [args.out_fpn_chan] * len(backbone.num_channels)
    elif args.sandwich_attn:
        attn_channels = (backbone.num_channels, [args.out_fpn_chan] * len(backbone.num_channels))
    else:
        attn_channels = backbone.num_channels
    attn = build_sa_layers(args, attn_channels)
    fpn = build_fpn(args, backbone.num_channels)
    head = build_head(args)
    
    model = NbmModel(
        args,
        backbone,
        attn,
        fpn,
        head
    ).to(device)

    model = initialize_model(model, path=os.path.join(mod_p, 'model_chkpt.pt'), train=False)

    return model, args


def postpro_detr(outputs, config, min_score=0.4):
    
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., 1:].max(-1)

    bs, n_queries = out_bbox.shape[:2]
    boxes = rel_to_coord(out_bbox.flatten(end_dim=1)).view(bs, n_queries, 4).to(torch.int64)
    
    output = []
    # Iterate batch and append final results
    for b_idx in range(bs):

        b_output = {}

        # Apply NMS separately for each class
        for class_idx in range(1, config.num_classes + 1): # class "other" ??
            class_where = (labels[b_idx] == class_idx - 1) & (scores[b_idx] > min_score)
            if not class_where.any():
                b_output[str(class_idx)] = dict(
                    bbox_coord=torch.Tensor(), 
                    scores=torch.Tensor())
                continue

            class_boxes = boxes[b_idx][class_where]#.cpu().numpy()
            class_scores = scores[b_idx][class_where].unsqueeze(0)#.cpu().numpy()

            b_output[str(class_idx)] = dict(
                bbox_coord=class_boxes,
                scores=class_scores
            )

        output.append(b_output)
    
    return output


def merge_images(fp, outputs, num_classes, nms_thresh=0.3):

    min_border_size = 0.9 * (fp.W_PIX - fp.HOP_SPECTRO)

    class_bbox = {}

    out = []
    for b_outputs in outputs:
        out.extend(b_outputs)

    range_min = 1
    range_max = num_classes + 1

    nms_bbox_inpt = []
    nms_scores_inpt = []
    nms_species = []

    for j in range(range_min, range_max):

        class_bbox[str(j)] = {}

        for i, img_out in enumerate(out):

            bbox_coord = img_out[str(j)]['bbox_coord'].unsqueeze(0)
            scores = img_out[str(j)]['scores']

            if len(img_out[str(j)]['bbox_coord']) == 0:
                continue

            # Remove boundary boxes that are entirely contained in the previous or following frame (prone to misclassification)
            widths = bbox_coord[..., 2] - bbox_coord[..., 0]

            if i == 0:
                condition = (bbox_coord[..., 2] >= fp.W_PIX - 5) & (widths < min_border_size)
            elif i == len(out) - 1:
                condition = (bbox_coord[..., 0] <= 4) & (widths < min_border_size)
            else:
                condition = ((bbox_coord[..., 0] <= 4) | (bbox_coord[..., 2] >= fp.W_PIX - 5)) & (widths < min_border_size)

            drop_idx = torch.nonzero(condition)[:, 1].cpu().numpy()
            keep_idx = [i for i in range(bbox_coord.size(1)) if i not in drop_idx]
            if len(keep_idx) == 0:
                continue
            bbox_coord = bbox_coord[:, keep_idx, :]
            scores = scores[:, keep_idx]

            bbox_coord[..., 0] += fp.HOP_SPECTRO * i
            bbox_coord[..., 2] += fp.HOP_SPECTRO * i

            # Now check that no bbox lies beyond file's end
            condition = bbox_coord[..., 2] >= fp.spectrogram_length
            drop_idx = torch.nonzero(condition)[:, 1].cpu().numpy()
            keep_idx = [i for i in range(bbox_coord.size(1)) if i not in drop_idx]
            if len(keep_idx) == 0:
                continue
            bbox_coord = bbox_coord[:, keep_idx, :]
            scores = scores[:, keep_idx]

            nms_bbox_inpt.append(bbox_coord)
            nms_scores_inpt.append(scores)
            nms_species += [j] * bbox_coord.size(1)

    if len(nms_bbox_inpt) == 0:
        for j in range(range_min, range_max):
            class_bbox[str(j)]['bbox_coord'] = torch.tensor([])
            class_bbox[str(j)]['scores'] = torch.tensor([])
    else:
        nms_bbox_inpt = torch.cat(nms_bbox_inpt, dim=1)
        nms_scores_inpt = torch.cat(nms_scores_inpt, dim=1)

        proposals, scores, nms_index = nms(nms_bbox_inpt, nms_scores_inpt, post_nms_topN=nms_bbox_inpt.shape[1], nms_thresh=nms_thresh, return_idx=True)
        species = np.array(nms_species)[nms_index[0]]
        proposals = proposals[0]
        scores = scores[0]

        for j in range(range_min, range_max):

            bird_idx = (species == j)

            if bird_idx.any() == False:
                class_bbox[str(j)]['bbox_coord'] = torch.tensor([])
                class_bbox[str(j)]['scores'] = torch.tensor([])
            else:
                class_bbox[str(j)]['bbox_coord'] = proposals[bird_idx]
                class_bbox[str(j)]['scores'] = scores[bird_idx]

    return class_bbox


def visualise_model_out(output, fp, spectrogram, reverse_dict, show_sp_name=True):

    time_limits = [(i * fp.HOP_SPECTRO, i * fp.HOP_SPECTRO + 1024) for i, _ in spectrogram]
    min_score = 0.01
    for idx, (i, spectro) in enumerate(spectrogram):

        print(i)
        start, end = time_limits[idx]

        fig, ax = plt.subplots(figsize=(16, 8))
        # Display the image
        ax.imshow(spectro.cpu().squeeze(), origin='lower')

        for b_id in np.arange(len(reverse_dict)):

            b_species = reverse_dict[b_id]

            if b_species not in output.keys():
                continue

            proposals = torch.Tensor(output[b_species]['bbox_coord'])
            class_scores = torch.Tensor(output[b_species]['scores'])

            idx = torch.nonzero(((start <= proposals[:, 0]) & (proposals[:, 0] < end)) |
                         ((start <= proposals[:, 2]) & (proposals[:, 2] < end)))[:, 0]
            bbox = proposals[idx]
            bbox[:, [0, 2]] = bbox[:, [0, 2]].clamp(min=start, max=end - 1)
            scores = class_scores[idx]

            for j, row in enumerate(bbox):

                x_1, y_1, x_2, y_2 = row
                x_1 = int(x_1 - start)
                x_2 = int(x_2 - start)

                # Create a Rectangle patch
                rect = patches.Rectangle((x_1, int(y_1)), x_2 - x_1, int(y_2) - int(y_1), linewidth=1, edgecolor='b', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                y_anchor = y_1 - 20
                if y_anchor < 10:
                    y_anchor = y_2 + 15
                score = np.round(scores[j].item(), 4)
                if score < min_score:
                    species = 'Unsure'
                else:
                    species = reverse_dict[b_id]

#                 Add the patch to the Axes
#                 ax.add_patch(rect)
                if show_sp_name:
                    ax.annotate(f'{species}, {score:.2f}', (x_1, y_anchor), backgroundcolor='b', color='white', fontsize='medium')
        pix_precision_y = 33.3
        pix_precision_x = 0.002993197278911565 # 0.003
        y_labels = [500 + int(y * pix_precision_y) for y in ax.get_yticks()]
        x_labels = [int(1000 * (x + i * 819) * pix_precision_x) / 1000 for x in ax.get_xticks()]
        ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks().tolist()))
        ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks().tolist()))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        plt.show()

