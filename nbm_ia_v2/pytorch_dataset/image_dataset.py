import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pickle
from scipy import signal


class Img_dataset(Dataset):
    
    def __init__(self, img_db_path=None, annotations_path=None, test_array=None, zero_max=False, normalize=False, noise_strength=0, transform=False, fold=0, test=False):
        super(Img_dataset, self).__init__()
        
        self.noise_strength = noise_strength
        self.zero_max = zero_max
        self.normalize = normalize
        self.transform = transform
        self.cross_val = fold > 0

        if img_db_path is not None:
            if type(img_db_path) == list:
                self.img_db = []
                for path in img_db_path:
                    with h5py.File(path, 'r') as f:
                        self.img_db.append(f['img_db'][...])
                self.lengths = [len(ds) for ds in self.img_db]
                self.lengths = np.cumsum(np.array(self.lengths))
                self.subs = [0] + list(self.lengths[:-1])
            else:
                with h5py.File(img_db_path, 'r') as f:
                    self.img_db = f['img_db'][...]
        elif test_array is not None:
            self.img_db = test_array
        else:
            print('Path to dataset or test file must be provided')
        
        # if no annotation is provided, it is a test dataset
        if annotations_path is not None:
            if type(annotations_path) == list:
                annotations_all = None
                for path in annotations_path:
                    with h5py.File(path, 'r') as f:
                        annotations = {
                            idx: {
                                'bb_coord': f[idx]['bb_coord'][...],
                                'birder': f[idx]['birder'][...],
                                'filename': f[idx]['filename'][...],
                                'bird_id': f[idx]['bird_id'][...]
                            } for idx in f.keys()
                        }
                    if annotations_all is None:
                        annotations_all = annotations
                    else:
                        annotations = {str(int(key) + len(annotations_all)): value for key, value in annotations.items()}
                        annotations_all.update(annotations)
                self.annotations = annotations_all
            else:
                with h5py.File(annotations_path, 'r') as f:
                    self.annotations = {
                        idx: {
                            'bb_coord': f[idx]['bb_coord'][...],
                            'birder': f[idx]['birder'][...],
                            'filename': f[idx]['filename'][...],
                            'bird_id': f[idx]['bird_id'][...]
                        } for idx in f.keys()
                    }
                
            # assert len(self.annotations) == len(self.img_db)

        else:
            self.annotations = None
        
        if self.cross_val:
            # Check if whole model of cross val training
            if fold == 4:
                with open('xc_test_files', 'rb') as f:
                    test_files = pickle.load(f)
                with open('test_files_extra_xc', 'rb') as f:
                    extra_test_files = pickle.load(f)
                test_files = np.concatenate([test_files, extra_test_files])
                if test:
                    self.authorized_idx = [idx for idx, val in self.annotations.items() if val['filename'] in test_files]
                else:
                    self.authorized_idx = [idx for idx, val in self.annotations.items() if val['filename'] not in test_files]
            else:
                with open('folds', 'rb') as f:
                    f_1, f_2, f_3 = pickle.load(f)
                which_fold = {1: f_1, 2: f_2, 3: f_3}
                if test:
                    self.authorized_idx = [idx for idx, val in self.annotations.items() if val['filename'] in which_fold[fold]]
                else:
                    self.authorized_idx = [idx for idx, val in self.annotations.items() if val['filename'] not in which_fold[fold]]

        
    def __len__(self):
        if self.cross_val:
            return len(self.authorized_idx)

        if type(self.img_db) == list:
            length = self.lengths[-1]
        else:
            length = len(self.img_db)
        return length


    def __getitem__(self, idx):
        if self.cross_val:
            idx = int(self.authorized_idx[idx])    

        if type(self.img_db) == list:
            db_idx = np.where(self.lengths > idx)[0][0]
            img = torch.Tensor(self.img_db[db_idx][idx - self.subs[db_idx]])
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
        else:
            img = torch.Tensor(self.img_db[idx])

        if self.noise_strength > 0:
            std = img.std().item()
            img += self.noise_strength * torch.clamp(torch.randn(img.size()), min=-2, max=2).mul_(std / 2)

        if self.zero_max:
            img -= img.max()

        if self.transform:

            # Random Gain
            img += np.random.uniform(0.05, 0.75)

            bool_transform = np.random.randint(2, size=3)

            # Random dynamic range compression
            if bool_transform[0] == 1:
                mean = img.mean()
                std = img.std()
                rdm_thresh = np.random.uniform(0.2, 1.2)
                rdm_gain = np.random.uniform(0.5, 1)
                mask = (img > mean + rdm_thresh * std).type(torch.float)
                img = img * rdm_gain * mask + img * (1 - mask)

            # Random low pass filter
            if bool_transform[1] == 1:
                freq_accuracy = 33.3
                cutting_freq = np.random.randint(500, 5000)
                b, a = signal.butter(1, 2 * np.pi * cutting_freq, 'low', analog=True)
                w, h = signal.freqs(b, a, worN=2 * np.pi * (500 + np.arange(256) * freq_accuracy))
                # Addition of gain in log space
                mat = torch.Tensor(20 * np.log10(np.clip(abs(h), 1e-9, None))).unsqueeze(0).repeat(1024, 1).transpose(0, 1).unsqueeze(0)
                img = img + mat

            # Random atmospheric absorption: air humidity and temperature are sampled randomly, affecting the frequency response
            # see https://www.mne.psu.edu/lamancusa/me458/10_osp.pdf
            if bool_transform[2] == 1:
                f = torch.Tensor(np.arange(500, 500 + 256 * 33.3, 33.3)[::-1].copy())
                f2 = f ** 2 
                # Sample relative humidity
                h_i, h_f = np.random.uniform(size=2)
                # Sample air temperature
                T_i, T_f = np.random.normal(size=2) * 4 + 285.15
                alpha_i = atm_abs_coeff(T_i, h_i, f2)
                alpha_f = atm_abs_coeff(T_f, h_f, f2)
                img = img + alpha_i - alpha_f

        if self.normalize:
            img = 0.5 + img / torch.abs(img.min())

        if self.annotations is not None:

            idx_annotations = self.annotations[str(idx)]
            bb_coord = torch.Tensor(idx_annotations['bb_coord'])
            bird_id = torch.Tensor(idx_annotations['bird_id']).int()
            img_info = (idx_annotations['birder'], idx_annotations['filename'])
        
            return (img, bb_coord, bird_id, img_info)

        else:
            return (img, torch.Tensor(), torch.Tensor(), tuple())


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