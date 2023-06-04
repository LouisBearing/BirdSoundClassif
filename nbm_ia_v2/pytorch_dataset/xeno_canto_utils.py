import os
from operator import itemgetter
import time
import json
import requests
import urllib
import pandas as pd
import numpy as np
from faster_rcnn.pytorch_dataset.utils import *
from faster_rcnn.ia_model_utils import *
from faster_rcnn.nets.faster_rcnn import *
from faster_rcnn.nets.faster_utils import *
from faster_rcnn.nets.layers import *
from faster_rcnn.nets.vgg_backbone import *
from faster_rcnn.pytorch_dataset.image_dataset import *
from faster_rcnn.pytorch_dataset.prepare_dataset import *
from faster_rcnn.pytorch_dataset.xeno_canto_utils import *
from torch.utils.data.sampler import SubsetRandomSampler


def download_request(species, sound_type, quality, max_length):

    # Load list of already processed file indexes

    with open(os.path.join(r'C:\Users\laeri\NBM\data_2\xc', 'file_ids.json'), 'r') as f:
        file_ids = json.load(f)
    file_ids = file_ids['file_ids']
    
    # XC API request    
    parameters = {
        'query': f'{species} type:"{sound_type}" len_lt: {max_length} q:{quality}'
    }

    response = requests.get('https://www.xeno-canto.org/api/2/recordings', params=parameters)
    js = response.json()

    filepath = r'C:\Users\laeri\NBM\data_2\xc\temp'

    for i in np.arange(len(js['recordings'])):
        recording = js['recordings'][i]
        rec_id = recording['id']
        if (rec_id in file_ids) or (recording['also'] != ['']) or ('juvenile' in recording['type']):
            continue
        elif (sound_type != 'song') and ('song' in recording['type']):
            continue
        filename = recording['gen'].lower() + '_' + recording['sp'].lower() + '_' + recording['id'] + '.mp3'
        urllib.request.urlretrieve(recording['file'], filename=os.path.join(filepath, filename))
        file_ids.append(rec_id)
        
    # Convert mp3 to wav
    dir_convert_mp32wav(filepath, keep_file=False)
        
    return file_ids


def process_temp_directory(bird_call_detection, b_idx, file_ids, delete=True):
    
    overlap_spectro = 0.4
    W_PIX = 1024
    HOP_SPECTRO = int((1 - overlap_spectro) * W_PIX)
    min_threshold = 15
    
    filepath = r'C:\Users\laeri\NBM\data_2\xc\temp'
    
    img_db = []
    all_bboxes = []
    bird_ids = []
    filenames = []
    
    # Evaluate detection model on all files in temp directory, save detected bounding boxes
    for file in os.listdir(filepath):
        class_bbox, spectrogram = bird_call_detection.process_wav(os.path.join(filepath, file), min_score=0.05)

        if (len(spectrogram) == 0) or (spectrogram.shape[1] < W_PIX):
            continue

        class_bbox = [class_bbox[str(i)]['bbox_coord'].cpu().numpy() for i in range(1, len(class_bbox))]
        non_empty_arrays = [bbox for bbox in class_bbox if len(bbox) > 0]

        if (len(non_empty_arrays) == 0):
            continue

        class_bbox = torch.Tensor(np.concatenate(non_empty_arrays))

        indexes = [(k * HOP_SPECTRO, k * HOP_SPECTRO + W_PIX) for k in np.arange(int((spectrogram.shape[1] - W_PIX) / HOP_SPECTRO) + 1)]
        
        # For each image
        for start, end in indexes:

            idx = torch.nonzero(((start <= class_bbox[:, 0]) & (class_bbox[:, 0] < end)) |
                         ((start <= class_bbox[:, 2]) & (class_bbox[:, 2] < end)))[:, 0]
            bbox = class_bbox[idx]
            bbox[:, [0, 2]] = bbox[:, [0, 2]].clamp(min=start, max=end - 1)

            bboxes = []

            for row in bbox:

                x_1, y_1, x_2, y_2 = row
                x_1 = int(x_1 - start)
                x_2 = int(x_2 - start)

                if ((x_2 - x_1 + 1) <= min_threshold) or ((y_2 - y_1 + 1) <= min_threshold):
                    continue

                bboxes.append([x_1, int(y_1), x_2, int(y_2)])

            if len(bboxes) > 0:
                img_db.append(np.array(spectrogram[:, start:end]))
                bboxes = np.stack(bboxes)
                all_bboxes.append(bboxes)
                bird_ids.append([b_idx] * len(bboxes))
                filenames.append(file)
                
    annotations = pd.DataFrame({'bbox_coord': all_bboxes, 'bird_id': bird_ids, 'filename': filenames})
    annotations['birder'] = 'XC'
    new_samples_count = np.array([len(n_sample) for n_sample in all_bboxes]).sum()
    
    # Delete downloaded files
    if delete:
        for file in os.listdir(filepath):
            os.remove(os.path.join(filepath, file))
            
    # Update file indexes list
    with open(os.path.join(r'C:\Users\laeri\NBM\data_2\xc', 'file_ids.json'), 'w') as f:
        json.dump({'file_ids': file_ids}, f)
    
    return img_db, annotations, new_samples_count


def refresh_file_list():
    with open(os.path.join(r'C:\Users\laeri\NBM\data_2\xc', 'file_ids.json'), 'w') as f:
        json.dump({'file_ids': []}, f)
        
def get_request(request=None, sound_type='nocturnal flight call'):
    
    if request is None:
        request = dict(quality='A', max_length=30 if (sound_type == 'nocturnal flight call') else 10, sound_type=sound_type)
        return request

    elif request['sound_type'] == 'nocturnal flight call':
        request = dict(quality='A', max_length=10, sound_type='flight call')
    
    elif request['quality'] != 'D':
        if (request['quality'] == 'C') or ((request['quality'] == 'B') and (request['max_length'] > 10)):
            request = dict(quality='A', max_length=request['max_length'] + 10, sound_type=request['sound_type'])
        else: 
            request = dict(quality=next_quality(request['quality']), max_length=request['max_length'], sound_type=request['sound_type'])
            
    if request['max_length'] > 20:
        return {}
    else:
        return request

def next_quality(quality):
    
    if quality == 'A':
        return 'B'
    if quality == 'B':
        return 'C'
    else:
        return 'D'


def fetch_more_samples(bird_call_detection, species, sound_type='nocturnal flight call', needed=500):
    
    dict_dir = r'C:\Users\laeri\NBM'
    with open(os.path.join(dict_dir, 'bird_dict.json'), 'r') as f:
        birds_dict = json.load(f)

    b_idx = birds_dict[species.capitalize()]

    if species == 'Luscinia megarhynchos megarhynchos':
        species = 'Luscinia megarhynchos'
    
    img_db = []
    annotations = pd.DataFrame()
    n_samples = 0
    
    request = get_request(sound_type=sound_type)
    print(f'Current request: {request}')
    
    while (needed > 0) and (len(request) > 0):
        
        print(f'Fetching {species} samples with request {request}, needed = {needed}')
        
        file_ids = download_request(species, **request)
        request_img_db, request_annotations, new_samples_count = process_temp_directory(bird_call_detection, b_idx, file_ids)
        
        img_db += request_img_db
        annotations = pd.concat([annotations, request_annotations])
        needed -= new_samples_count
        n_samples += new_samples_count
        
        if needed > 0:
            request = get_request(request, sound_type=sound_type)
            
    return img_db, annotations, n_samples


def order_reciped_XC(bird_call_detection, recipe, n_output_files=6):
    
    img_db = []
    annotations = pd.DataFrame()
    
    for species, values in recipe.items():
        bird_img_db, bird_annotations, n_samples = fetch_more_samples(bird_call_detection, species, values['sound_type'], 
                                                                      values['needed'])
        img_db += bird_img_db
        annotations = pd.concat([annotations, bird_annotations])
        print(f'Got {n_samples} samples of {species} {values["sound_type"]}')
        
    # Shuffle files
    indexes = np.arange(len(img_db))
    np.random.shuffle(indexes)
    indexes = list(indexes)

    img_db = np.stack(itemgetter(*indexes)(img_db))
    annotations.index = range(len(annotations))
    annotations = annotations.loc[indexes]
    
    # Serialize in all datasets

    filepath = r'C:\Users\laeri\NBM\data_2\xc\database'
    samples_per_file = len(img_db) // n_output_files

    for i in range(n_output_files):

        with h5py.File(os.path.join(filepath, f'img_db_{i}.hdf5'), 'w') as f:
            f.create_dataset('img_db', data=img_db[i * samples_per_file:(i + 1) * samples_per_file])

        short_annotations = annotations.iloc[i * samples_per_file:(i + 1) * samples_per_file]
        with h5py.File(os.path.join(filepath, f'annotations_{i}.hdf5'), 'w') as f:

            for idx in range(len(short_annotations)):

                grp = f.create_group(str(idx))
                subds = short_annotations.iloc[idx]
                bb_coord = np.vstack(subds.bbox_coord)
                grp.create_dataset('bb_coord', data=bb_coord)

                for key in ['bird_id', 'filename', 'birder']:
                    grp.create_dataset(key, data=subds[key])