import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches

from src.models.util.nets_utils import nms


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


def visualise_model_out(output, fp, spectrogram, reverse_dict):

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
