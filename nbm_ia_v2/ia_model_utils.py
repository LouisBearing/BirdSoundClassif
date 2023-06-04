import os
import argparse
import numpy as np
import json
import time

from .nets.faster_rcnn import *
from .nets.faster_utils import *
from .nets.layers import *
from .nets.vgg_backbone import *
from .pytorch_dataset.image_dataset import *
from .pytorch_dataset.prepare_dataset import *



def load_model(save_dir, post_nms_topN_eval=50, biophonia=True, device=None):

    config = Config()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Args
    args = {}
    args_path = os.path.join(save_dir, 'args')
    with open(args_path, 'rb') as f:
        args = json.load(f)
            
    for attr, attr_value in args.items():
        setattr(config, attr, attr_value)
        
    setattr(config, 'post_nms_topN_eval', post_nms_topN_eval)
    setattr(config, 'device', device)
    if biophonia:
        setattr(config, 'use_biophonia', True)
    
    vgg = vgg16_bn(pretrained=False, input_channels=config.input_channels,
        fpn=config.fpn, fpn_p_channels=config.fpn_p_channels, self_attention=config.self_attention, encode_frequency=config.encode_frequency).to(device)
    model = Faster_RCNN(config, vgg).to(device)
    epoch = model.resume(save_dir)
    model.eval()

    ia_wrapper = ia_model(model, config)
    
    return ia_wrapper


class ia_model():

    def __init__(self, model, config):

        self.model = model
        self.config = config

    def process_wav(self, wav_path, batch_size=4, min_score=0.5, return_spectrogram=False):
    
        directory = os.path.dirname(wav_path)
        f = os.path.basename(wav_path)

        # File preprocessing
        print('Generating spectrograms... This step can take some time')
        t = time.time()
        self.fp = File_Processor(directory, f)
        img_db = self.fp.process_file(overlap_spectro=0.2, w_pix=1024)
        temp_res = self.fp.DT
        print(f'Résolution temporelle: 1pix = {temp_res}s')
        print(f'File {os.path.basename(wav_path)} successfully processed in {int(time.time() - t)} sec. \n **** \n Automatic bird sound detection....')
        
        # If img db empty, ex if wrong sampling frequency
        if len(img_db) == 0:
            return np.array([]), np.array([]), 0.003
        
        img_db = np.stack(img_db)

        # Dataset
        img_dataset = Img_dataset(test_array=img_db, zero_max=self.config.zero_max, normalize=self.config.normalize_input)
        loader = DataLoader(img_dataset, batch_size=batch_size, collate_fn=collate_fn)

        loader_length = len(loader)
        if loader_length > 2:
            verbose = True
            if loader_length > 100:
                step = int(loader_length / 20)
            elif loader_length > 20:
                step = int(loader_length / 10)
            else:
                step = int(loader_length / 3)
            steps = list(np.arange(loader_length, step=step))
        else:
            verbose = False
        
        # Outputs & spectrogram
        t = time.time()
        outputs = []
        spectrogram = []
        for i, batch in enumerate(iter(loader)):
            if verbose:
                if len(steps) > 0 and i > steps[0]:
                    remaining_time_est = int(t - time.time() + (time.time() - t) / (i / loader_length))
                    print(f'{int(100 * i / loader_length)} % processed... \n Remaining time estimation: {remaining_time_est // 60} min, {remaining_time_est % 60} sec \n')
                    steps.pop(0)
            img, foe, bar, _, _ = batch
            # if self.config.normalize_input:
            #     max_power = img.max(dim=(-1)).values.max(dim=-1).values
            #     img = img - max_power.reshape((len(img), 1, 1, 1))
            batch_out = self.model(img, evaluation=False, min_score=min_score)
            outputs.append(batch_out)
            if return_spectrogram:
                for sample_id, sample in enumerate(batch_out):
                    boxes = [sample[str(b_id)]['bbox_coord'] for b_id in np.arange(1, len(sample)) if len(sample[str(b_id)]['bbox_coord'] > 0)]
                    if len(boxes) > 0:
                        idx = i * batch_size + sample_id
                        spectrogram.append((idx, img[sample_id]))
        print(f'~~~~~~ File successfully processed in {int(time.time() - t)} seconds ~~~~~~~')

        # if spectrogram[0].shape[0] == 1:
        #     return np.array([]), np.array([])

        # if return_spectrogram:
        #     spectrogram = torch.cat(spectrogram, dim=0)
        #     if spectrogram.shape[0] > 1:
        #         spectrogram = torch.cat([
        #             torch.cat([spectrogram[i, ..., :fp.HOP_SPECTRO] for i in range(spectrogram.size(0) - 1)], dim=-1).squeeze(),
        #             spectrogram[-1].squeeze()
        #         ], dim=-1)[:, :fp.spectrogram_length]
        #     else:
        #         spectrogram = spectrogram.squeeze()
        
        class_bbox = self.merge_images(self.fp, outputs)
        
        return class_bbox, spectrogram, temp_res
        
        
    def merge_images(self, fp, outputs, nms_thresh=0.3):
        
        min_border_size = 0.9 * (fp.W_PIX - fp.HOP_SPECTRO)

        class_bbox = {}

        out = []
        for b_outputs in outputs:
            out.extend(b_outputs)

        if self.config.num_classes == 1:
            range_min = 1
            range_max = 2
        else:
            range_min = 0
            range_max = self.config.num_classes
            
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