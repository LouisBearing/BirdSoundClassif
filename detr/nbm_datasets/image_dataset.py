import os
import torch
import glob
import numpy as np
import pandas as pd
import imageio
from scipy import signal
from torch.utils.data import Dataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class Img_dataset(Dataset):
    
    def __init__(self, dataset_path, transform=False):
        super(Img_dataset, self).__init__()
        
        self.ds_p = dataset_path
        self.transform = transform

        self.positive_files = []
        for f in os.listdir(os.path.join(self.ds_p, 'positive_files')):
            self.positive_files.extend([os.path.basename(img) for img in glob.glob(os.path.join(self.ds_p, 'positive_files', f) + '/*.png')])
        self.negative_files = []
        for f in os.listdir(os.path.join(self.ds_p, 'negative_files')):
            self.negative_files.extend([os.path.basename(img) for img in glob.glob(os.path.join(self.ds_p, 'negative_files', f) + '/*.png')])
        self.hard_negative_files = []
        for f in os.listdir(os.path.join(self.ds_p, 'hard_neg')):
            self.hard_negative_files.extend([os.path.basename(img) for img in glob.glob(os.path.join(self.ds_p, 'hard_neg', f) + '/*.png')])

        
    def __len__(self):
        return len(self.positive_files)


    def __getitem__(self, idx):

        imgp = self.positive_files[idx]
        splits = imgp.replace('.png', '').split('__')
        file, fileidx = '__'.join(splits[:-1]), splits[-1]

        # Load image
        img = imageio.imread(os.path.join(self.ds_p, 'positive_files', file, imgp))
        img = torch.Tensor(img / 255)

        # Load annots
        annotp = os.path.join(self.ds_p, 'positive_files', file, 'annotations.csv')
        annot = pd.read_csv(annotp, sep=';')
        annot['coord'] = annot['coord'].apply(eval)
        annot['bird_id'] = annot['bird_id'].apply(eval)
        bboxes, bird_ids = annot.loc[annot['index'] == int(fileidx), ['coord', 'bird_id']].values[0]
        bboxes, bird_ids = torch.Tensor(bboxes), torch.Tensor(bird_ids)

        # Negative sample
        negp = np.random.choice(self.negative_files, 1)[0]
        splits = negp.replace('.png', '').split('__')
        neg_file = '__'.join(splits[:-1])
        neg_img = imageio.imread(os.path.join(self.ds_p, 'negative_files', neg_file, negp))
        neg_img = torch.Tensor(neg_img / 255)

        if self.transform:
            std = img.std().item()
            noise = torch.clamp(torch.randn(img.shape).mul_(img.std().item() / 2), min=-0.5, max=0.5)
            # Random Gain
            img += np.random.uniform(-0.1, 0.35)
            img += noise

            bool_transform = np.random.randint(2, size=4)
            if bool_transform[0] == 1:
                hard_negp = np.random.choice(self.hard_negative_files, 1)[0]
                splits = hard_negp.replace('.png', '').split('__')
                hard_neg_file = '__'.join(splits[:-1])
                hard_neg_img = imageio.imread(os.path.join(self.ds_p, 'hard_neg', hard_neg_file, hard_negp))
                hard_neg_img = torch.Tensor(hard_neg_img / 255)
                ## add to positive img
                coef = np.random.uniform(0.1, 0.4)
                img = (img + coef * hard_neg_img) / (1 + coef)
                ## add to negative img
                neg_coef = np.random.uniform(0.5, 0.99)
                neg_img = (neg_img + neg_coef * hard_neg_img) / (1 + neg_coef)

            # # Random Gain
            # img += np.random.uniform(-0.5, 0.5)

            # bool_transform = np.random.randint(2, size=3)

            # # Random dynamic range compression
            # if bool_transform[0] == 1:
            #     mean = img.mean()
            #     std = img.std()
            #     rdm_thresh = np.random.uniform(0.2, 1.2)
            #     rdm_gain = np.random.uniform(0.5, 1)
            #     mask = (img > mean + rdm_thresh * std).type(torch.float)
            #     img = img * rdm_gain * mask + img * (1 - mask)

            # # Random low pass filter
            # if bool_transform[1] == 1:
            #     freq_accuracy = 33.3
            #     cutting_freq = np.random.randint(500, 5000)
            #     b, a = signal.butter(1, 2 * np.pi * cutting_freq, 'low', analog=True)
            #     w, h = signal.freqs(b, a, worN=2 * np.pi * (500 + np.arange(256) * freq_accuracy))
            #     # Addition of gain in log space
            #     mat = torch.Tensor(20 * np.log10(np.clip(abs(h), 1e-9, None))).unsqueeze(0).repeat(1024, 1).transpose(0, 1).unsqueeze(0)
            #     img = img + mat

            # # Random atmospheric absorption: air humidity and temperature are sampled randomly, affecting the frequency response
            # # see https://www.mne.psu.edu/lamancusa/me458/10_osp.pdf
            # if bool_transform[2] == 1:
            #     f = torch.Tensor(np.arange(500, 500 + 256 * 33.3, 33.3)[::-1].copy())
            #     f2 = f ** 2 
            #     # Sample relative humidity
            #     h_i, h_f = np.random.uniform(size=2)
            #     # Sample air temperature
            #     T_i, T_f = np.random.normal(size=2) * 4 + 285.15
            #     alpha_i = atm_abs_coeff(T_i, h_i, f2)
            #     alpha_f = atm_abs_coeff(T_f, h_f, f2)
            #     img = img + alpha_i - alpha_f

        return (img, neg_img, bboxes, bird_ids)


def atm_abs_coeff(T, h, f2):
    T_0 = 293.15
    Fr_O = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    Fr_N = ((T_0 / T) ** 0.5) * (9 + 280 * h * np.exp(-4.17 * (-1 + (T_0 / T) ** (1 / 3))))
    alpha = 869 * f2 * (1.84e-11 * ((T / T_0) ** 0.5) + ((T_0 / T) ** 2.5) * (0.01275 * np.exp(-2239.1 / T) / (Fr_O + f2 / Fr_O) + 0.1068 * np.exp(-3352 / T) / (Fr_N + f2 / Fr_N)))
    return alpha.unsqueeze(0).repeat(1024, 1).transpose(0, 1).unsqueeze(0)


def plot_img_bb(img_dataset, img_idx, show_bb=True, channel=0):
    """
    Takes as input an Img_dataset instance and plots the spectrogram corresponding to the given index together with bird calls bounding boxes
    """
    
    img_array, bb_coord, img_info = img_dataset[img_idx]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8,4))

    # Display the image
    ax.imshow(img_array[channel, ...], origin='lower')
    
    coord_list = bb_coord.tolist()
    
    if show_bb:
    
        for (x_1, y_1, x_2, y_2) in coord_list:

            # Create a Rectangle patch
            rect = patches.Rectangle((int(x_1), int(y_1)), int(x_2) - int(x_1), int(y_2) - int(y_1), linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.title(f'{img_info[0]}_{img_info[1]}')

    plt.show()