import numpy as np
import os
import torch
import json
from tqdm import tqdm
# from detr.nbm_datasets.image_dataset import *
from src.features.prepare_dataset import *
from src.models.detr import *
from src.models import build_model
from src.models.util.box_ops import *
from src.models.util.nets_utils import *


device = 'cpu'


def run_detection(model, config, wav_path, min_score=0.5, bs=10, return_spectrogram=True):
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
    device = 'cpu'
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
            batch = torch.Tensor(np.stack(batch)) # .to(device)
            with torch.no_grad():
                o = model(batch[:, None])
            batch_out = postpro_detr(o, config, min_score=min_score)
            outputs.append(batch_out)
            
            if return_spectrogram:
                for sample_id, sample in enumerate(batch_out):
                    boxes = [sample[str(b_id)]['bbox_coord'] for b_id in np.arange(1, len(sample)) if len(sample[str(b_id)]['bbox_coord'] > 0)]
                    if len(boxes) > 0:
                        idx = b_idx * bs + sample_id
                        spectrogram.append((idx, batch[sample_id]))  
            batch = []
            b_idx += 1

    return fp, outputs, spectrogram


def load_model(mod_p):

    args_path = os.path.join(mod_p, 'args')
    with open(args_path, 'rb') as f:
        args = json.load(f)

    config = Config()
    for attr, attr_value in args.items():
        setattr(config, attr, attr_value)

    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = DETR(
        backbone,
        transformer,
        num_classes=config.num_classes,
        num_queries=config.num_queries,
        aux_loss=False,
    )

    model = load_weights_cpu(config, model, path=os.path.join(mod_p, 'model_chkpt_last.pt'), train=False) # .to(config.device)

    return model, config


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
