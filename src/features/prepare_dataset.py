import numpy as np
import os
import time
from .utils import *
import librosa
import soundfile
import glob
import pickle
import imageio
import shutil


ornithos = {
    'NidalIssa': {
        'extra_label': ''
    },
    'KevinLeveque': {
        'extra_label': ''
    },
    'HerveRenaudineau': {
        'extra_label': ''
    },
    'GuillaumeBigayon': {
        'extra_label': ''
    },
    'GhislainRiou': {
        'extra_label': ''
    },
    'GaëtanMineau': {
        'extra_label': ''
    },
    'FredericCazaban': {
        'extra_label': ''
    },
    'ChristopheMercier': {
        'extra_label': ''
    },
    'AymericMousseau': {
        'extra_label': 'amousseau_'
    },
    'AdrienPajot': {
        'extra_label': ''
    },
    'WillyRaitiere': {
        'extra_label': 'willyraitiere_'
    },
    'MaxencePajot': {
        'extra_label': 'Piste de marqueur'
    },
    'MathurinAubry': {
        'extra_label': ''
    },
    'LionelManceau': {
        'extra_label': ''
    },
    'mediae': {
        'extra_label': ''
    }
}

keywords = [
    'anthus_pratensis',
    'apus_apus',
    'ardea_cinerea',
    'calidris_alpina',
    'charadrius_morinellus',
    'numenius_arquata',
    'tyto_alba',
    'vanellus_vanellus',
    'fringilla_coelebs#444457',
    'fringilla_coelebs#781870',
    'linaria_cannabina#606298',
    'rallus_aquaticus#789124',
    'rallus_aquaticus#794338'
]


def prepare_dataset(directory, out_directory, freq_accuracy=33.3, dt=0.003, overlap_spectro=0.2, w_pix=1024, annotations=True, audio_format=''):
    """
    Process all audio files in a directory, then save spectrogram and annotations in the destination directory
    """
    top_dir = directory.split('\\')[-1]
    extra_str_label = ornithos[top_dir]['extra_label'] if top_dir in ornithos.keys() else ''
    if audio_format != '':
        audio_files = glob.glob(os.path.joint(directory, "audio", f'*.{audio_format}'))
    else:
        audio_files = glob.glob(os.path.join(directory, "audio", '*.wav')) + glob.glob(os.path.join(directory, "audio", '*.mp3'))
    if annotations:
        labels = create_label_dataset(directory, extra_str_label=extra_str_label, suppress_unID=True, is_csv=top_dir == 'mediae')
    else:
        labels = None

    for file in audio_files:

        fp = File_Processor(file, extra_str_label, labels)
        out_pos_dir = os.path.join(out_directory, 'positive_files', top_dir + '__' + fp.filename.replace('#', '__'))
        out_neg_dir = os.path.join(out_directory, 'negative_files', top_dir + '__' + fp.filename.replace('#', '__'))
        if os.path.exists(out_pos_dir) or os.path.exists(out_neg_dir):
            continue

        print(f'~~~ Processing file {fp.filename} ~~~')
        img_db, annotations = fp.process_file(freq_accuracy=freq_accuracy, dt=dt, overlap_spectro=overlap_spectro, w_pix=w_pix)
        if img_db is None:
            continue
        if annotations is None:
            pos_idx = []
            n_img = len(img_db) # one case, we don't expect long noise file for dataset building
        elif type(annotations) == list:
            lengths = [len(e) for e in img_db]
            lengths = np.cumsum([0] + lengths)
            n_img = lengths[-1]
            pos_idx = [annot['index'].values for annot in annotations]
            pos_idx = np.concatenate([idx + inc for idx, inc in zip(pos_idx, lengths)]).astype(int)
            annotations = pd.concat(annotations)
            annotations['index'] = pos_idx
        else:
            pos_idx = annotations['index'].values
            n_img = len(img_db)

        if len(pos_idx) > 0:
            os.makedirs(out_pos_dir, exist_ok=True)
            annotations.to_csv(os.path.join(out_pos_dir, 'annotations.csv'), sep=';', index=False)
        if len(pos_idx) < n_img:
            os.makedirs(out_neg_dir, exist_ok=True)

        for i in range(n_img):
            if type(img_db[0]) == list:
                bin_number = (lengths <= i).sum() - 1
                bin_idx = i - lengths[bin_number]
                img = img_db[bin_number][bin_idx]
            else:
                img = img_db[i]
            file_idx = '__'.join([top_dir, fp.filename.replace('#', '__'), format(i, '05d')]) + '.png'
            img = np.round(img * 255).astype(np.uint8)
            if i in pos_idx:
                imageio.imwrite(os.path.join(out_pos_dir, file_idx), img)
            elif i <= 999:
                imageio.imwrite(os.path.join(out_neg_dir, file_idx), img)


class File_Processor:
    
    ### Parameters definition
    
    H_PIX = 375 # px
    LOW_FREQ = 500 # hz
    FREQ = 44100 # sampling rate, hz
    
    def __init__(self, filepath, extra_str_label='', labels=None):
        
        self.labels = labels
        self.ext = os.path.basename(filepath).split('.')[-1]
        self.filename = os.path.basename(filepath).replace('.' + self.ext, '').replace(extra_str_label, '')
        self.filepath = filepath
    
    
    def process_file(self, freq_accuracy=33.3, dt=0.003, overlap_spectro=0.2, w_pix=1024):
        '''
        Generates and split spectrogram into images of chosen width, and associate labels to each image under the form of bounding box coordinates
        '''

        # Final images
        self.W_PIX = w_pix
        self.HOP_SPECTRO = int((1 - overlap_spectro) * self.W_PIX)

        # Generate spectrogram
        data = self.load()
        if data is None:
            return None, None
        long_file_out = self.process_long_file(data, freq_accuracy, dt, overlap_spectro)
        if long_file_out is not None:
            return long_file_out

        self.WIN_LENGTH = int(self.FREQ / freq_accuracy)
        self.HOP_LENGTH = int(self.FREQ * dt)
        overlap_fft = np.round(1 - self.HOP_LENGTH / self.WIN_LENGTH, 3)

        # Actual hop duration & frequence accuracy
        self.FREQ_ACCURACY = self.FREQ / self.WIN_LENGTH
        self.DT = int((1 - overlap_fft) * self.WIN_LENGTH) / self.FREQ

        # Cut low and high freq
        self.LOW_IDX = 1 + int(self.LOW_FREQ / self.FREQ_ACCURACY)
        self.HIGH_IDX = self.LOW_IDX + self.H_PIX

        self.LOW_FREQ = (self.LOW_IDX - 1) * self.FREQ_ACCURACY
        self.HIGH_FREQ = (self.HIGH_IDX - 1) * self.FREQ_ACCURACY

        power_spec = self.spectrogram(data)

        # Record the length of the spectrogram
        self.spectrogram_length = sum(np.array([e.shape[-1] for e in power_spec]))

        # images to append
        img_db = self.split_power_spec(power_spec)

        # labels to append
        if self.labels is not None:
            try:
                labels_ = self.merge_and_filter_labels(img_db)
            except pd.errors.IntCastingNaNError:
                print('Something went wrong with the annotation file, skipping~~')
                return None, None
            return img_db, labels_
        else:
            return img_db, None

    
    def load(self):
        try:
            data, sr = librosa.core.load(self.filepath, sr=None)
        except:
            print('File loading failed')
            return
        if sr != self.FREQ:
            if ' ' in self.filename:
                norm_filename = self.filepath.replace(' ', '')
                shutil.copyfile(self.filepath, norm_filename)
            else:
                norm_filename = self.filepath
            temp_f = f'temp.{self.ext}'
            if self.ext == 'wav':
                command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 44100 %s" % (norm_filename, temp_f)
            else:
                command = "ffmpeg -i %s -ar 44100 %s" % (norm_filename, temp_f)
            os.system(command)
            data, _ = librosa.core.load(temp_f, sr=None) # wavfile.read(temp_audio)
            os.remove(temp_f)
            if ' ' in self.filename:
                os.remove(norm_filename)

        return data


    def process_long_file(self, data, freq_accuracy, dt, overlap_spectro):
        '''
        If file length exceeds the hardcoded max_l length, then it is split into chunks that are processed successively.
        '''

        output = None
        # If the file is too long, process in several steps
        max_l = int(15e7) - int(15e7) % self.FREQ
        if len(data) > max_l:
            print('Long file, processing in several steps...')
            for k in range(int(len(data) / max_l) + 1):
                outp = f'temp{str(k)}.{self.ext}'
                soundfile.write(outp, data[k * max_l: (k + 1) * max_l], self.FREQ)
            print('Done splitting input file')
            img_db = []
            annotations = []
            time_increment = max_l / self.FREQ
            for k in range(int(len(data) / max_l) + 1):
                print(f'~~ Processing split # {k} ~~')
                if self.labels is not None:
                    labels = self.labels.loc[self.labels['filename'] == self.filename].copy()
                    for col in ['t_start', 't_end']:
                        labels[col] = labels[col] - k * time_increment
                    labels = labels.loc[labels['t_start'].between(0, time_increment)]
                    labels['t_end'].clip(upper=time_increment, inplace=True)
                    labels['filename'] = f'temp{str(k)}'
                else:
                    labels = None
                fp = File_Processor(f'temp{str(k)}.{self.ext}', '', labels)
                img_db_inc, annotations_inc = fp.process_file(freq_accuracy=freq_accuracy, dt=dt, overlap_spectro=overlap_spectro, w_pix=self.W_PIX)
                img_db.append(img_db_inc)
                annotations.append(annotations_inc)
                os.remove(f'temp{str(k)}.{self.ext}')
            output = (img_db, annotations)
        
        return output


    def amp_to_db(self, x, min_level_db=-100):
        min_level = np.exp(min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    
    def spectrogram(self, data):
        max_l = int(5e7)
        stfts = []
        for k in range(int(len(data) / max_l) + 1):
            stfts.append(librosa.stft(data[k * max_l: (k + 1) * max_l], n_fft=self.WIN_LENGTH, hop_length=self.HOP_LENGTH))
        # stft = np.concatenate([self.amp_to_db(np.abs(stft)) for stft in stfts], axis=1)

        stfts = [self.amp_to_db(np.abs(stft)) for stft in stfts]

        ## Normalize
        # spectrogram = stft[self.LOW_IDX:self.HIGH_IDX, :]
        # s_max = spectrogram.max()
        # s_min = spectrogram.min()
        # spectrogram = ((spectrogram - s_min) / (s_max - s_min))
        stfts = [stft[self.LOW_IDX:self.HIGH_IDX, :] for stft in stfts]
        s_max = max([stft.max() for stft in stfts])
        s_min = min([stft.min() for stft in stfts])
        stfts = [(stft - s_min) / (s_max - s_min) for stft in stfts]

        return stfts
    
    
    def split_power_spec(self, log_power_spec):
        """
        Splits a spectrogram 2D array along axis=1 given hop size and img width.
        """

        # Split into overlapping fixed size images
        # img_db = [log_power_spec[:, k * self.HOP_SPECTRO: k * self.HOP_SPECTRO + self.W_PIX] for k in range(max(1, 
        # int(1 + np.ceil((log_power_spec.shape[-1] - self.W_PIX) / self.HOP_SPECTRO))))]
        lengths = [e.shape[-1] for e in log_power_spec]
        lengths = np.cumsum([0] + lengths)
        max_l = lengths[-1]
        img_db = []
        for k in range(max(1, int(1 + np.ceil((max_l - self.W_PIX) / self.HOP_SPECTRO)))):
            start_idx = k * self.HOP_SPECTRO
            end_idx = k * self.HOP_SPECTRO + self.W_PIX
            s_bin = (start_idx >= lengths).sum() - 1
            s_bin_idx = start_idx - lengths[s_bin]
            e_bin = (end_idx > lengths).sum() - 1
            e_bin_idx = end_idx - lengths[e_bin] if (e_bin < len(lengths) - 1) else None
            next_bin = (e_bin > s_bin) and (e_bin < len(lengths) - 1)
            if next_bin:
                img_db.append(np.concatenate([log_power_spec[s_bin][:, s_bin_idx:], log_power_spec[e_bin][:, :e_bin_idx]], axis=1))
            else:
                img_db.append(log_power_spec[s_bin][:, s_bin_idx:e_bin_idx])

        if img_db[-1].shape[-1] < self.W_PIX:

            if (self.labels is not None) and len(self.labels.loc[self.labels['filename'] == self.filename]) > 0:
                max_pix = int(self.labels.loc[self.labels['filename'] == self.filename, 't_end'].max() / self.DT)
            else:
                max_pix = max_l - self.W_PIX
            empty_width = max_l - max_pix

            while img_db[-1].shape[-1] < self.W_PIX:
                pad_width = max(1, min(empty_width, self.W_PIX - img_db[-1].shape[-1]))
                img = np.pad(img_db[-1], ((0, 0), (0, pad_width)), mode='reflect')
                img_db[-1] = img
                empty_width += pad_width

        return img_db
    
    
    def merge_and_filter_labels(self, img_db):
        """
        Computes and return a dataframe containing img indexes and a list of bb coordinates for each images in a given file
        """

        # Img coordinates in original spectrogram
        img_coord = [(i * self.HOP_SPECTRO, i * self.HOP_SPECTRO + self.W_PIX - 1) for i in range(len(img_db))]
        img_coord = pd.DataFrame(img_coord).rename(columns={0: 'start', 1:'end'})

        # Merge filtered label dataset with each image in collection, keep only annotations that intersect the images
        labels_ = self.labels.loc[self.labels['filename'] == self.filename].copy()
        # if mp3 file, suppress offset added in audacity, here this is a hardcoded 0.025s
        if self.ext == 'mp3':
            if not np.array([k in self.filename for k in keywords]).any():
                for col in ['t_start', 't_end']:
                    labels_[col] = labels_[col] - 0.03
        
        if len(labels_) == 0:
            # labels_ = pd.DataFrame({key: [] for key in ['index', 'coord', 'bird_id']})
            raise pd.errors.IntCastingNaNError
            # return labels_

        # Convert second to pixels given DT, the time equivalent of hop_size
        for ex_label, new_label in zip(['t_start', 't_end'], ['x_1', 'x_2']):
            labels_[new_label] = (labels_[ex_label].astype(float) / self.DT).astype(int)

        # Same for frequencies
        for ex_label, new_label in zip(['f_start', 'f_end'], ['y_1', 'y_2']):
            labels_[new_label] = ((labels_[ex_label].clip(lower=self.LOW_FREQ, upper=self.HIGH_FREQ) - self.LOW_FREQ) / self.FREQ_ACCURACY).astype(int)

        labels_ = labels_.loc[labels_['y_1'] != labels_['y_2']]
        labels_.index = range(len(labels_))

        labels_['w'] = labels_['x_2'] - labels_['x_1'] + 1
        labels_['h'] = labels_['y_2'] - labels_['y_1'] + 1

        for size in ['w', 'h']:
            labels_ = labels_.loc[labels_[size] > 0]

        labels_['joint'] = 1
        img_coord['joint'] = 1
        img_coord.reset_index(inplace=True)

        coord = ['x_1', 'y_1', 'x_2', 'y_2']
        labels_ = labels_[coord + ['w', 'h', 'joint', 'bird_id']].merge(img_coord, on='joint')
        labels_ = labels_.loc[(labels_['x_1'].between(labels_['start'], labels_['end'])) | (labels_['x_2'].between(labels_['start'], labels_['end'])) \
            | (labels_['x_1'].lt(labels_['start']) & labels_['x_2'].gt(labels_['end']))]

        # Supress bbox with too small intersection with spectrogram
        labels_['inside'] = labels_[['x_2', 'end']].min(axis=1) - labels_[['x_1', 'start']].max(axis=1) + 1

        cond_1 = (labels_['inside'] < 0.5 * labels_['w']) & (labels_['inside'] < 20)
        cond_2 = (labels_['inside'] < 0.1 * labels_['w']) & (labels_['inside'] < 45)

        labels_ = labels_.loc[~(cond_1 | cond_2)]

        # Bounding boxes are expanded 10% in every direction
        # labels_['x_1'] = (labels_['x_1'] - labels_['start'] - (labels_['w'] * 0.1).astype(int).clip(lower=3, upper=6)).clip(lower=0)
        # labels_['x_2'] = (labels_['x_2'] - labels_['start'] + (labels_['w'] * 0.1).astype(int).clip(lower=3, upper=6)).clip(upper=self.W_PIX - 1)
        # labels_['y_1'] = (labels_['y_1'] - (labels_['h'] * 0.1).astype(int).clip(lower=3, upper=6)).clip(lower=0)
        # labels_['y_2'] = (labels_['y_2'] + (labels_['h'] * 0.1).astype(int).clip(lower=3, upper=6)).clip(upper=self.H_PIX - 1)

        labels_['x_1'] = (labels_['x_1'] - labels_['start']).clip(lower=0)
        labels_['x_2'] = (labels_['x_2'] - labels_['start']).clip(upper=self.W_PIX - 1)
        labels_['y_1'] = (labels_['y_1']).clip(lower=0)
        labels_['y_2'] = (labels_['y_2']).clip(upper=self.H_PIX - 1)

        labels_['w'] = labels_['x_2'] - labels_['x_1']
        labels_['h'] = labels_['y_2'] - labels_['y_1']

        labels_['coord'] = [(x_1, y_1, x_2, y_2) for (x_1, y_1, x_2, y_2) in zip(labels_['x_1'], labels_['y_1'],
                                                                             labels_['x_2'], labels_['y_2'])]

        # Delete negative samples if they appear in a positive image
        labels_ = labels_.merge(labels_.loc[labels_['bird_id'] != -1].groupby('index').size().reset_index().rename(columns={0: 'count'}), on='index')
        labels_ = labels_.loc[(labels_['bird_id'] != -1) | (labels_['count'] == 0)]                                                 

        # One row per img
        labels_ = labels_.groupby('index', as_index=False).agg({'coord': lambda x: x.tolist(), 'bird_id': lambda x: x.tolist()})

        return labels_