import os
import json
import requests
import urllib
import ffmpeg
import numpy as np
import pandas as pd
import glob
import argparse
import ssl
from tqdm import tqdm


def download_request(args):

    # Load list of already processed file indexes

    file_ids_path = os.path.join(args.filepath, 'file_ids.json')
    if os.path.isfile(file_ids_path):
        try:
            with open(file_ids_path, 'r') as f:
                file_ids = json.load(f)
        except:
            file_ids = []
    else:
        file_ids = []
    
    # XC API request    
    parameters = {
        'query': f'{args.species} type:"{args.sound_type}" len_lt: {args.max_length} q:{args.quality}'
    }
    # TODO: permettre de ne pas préciser la qualité

    write_path = os.path.join(args.filepath, '_'.join(args.species.split()))
    os.makedirs(write_path, exist_ok=True)

    response = requests.get('https://www.xeno-canto.org/api/2/recordings', params=parameters)
    js = response.json()

    request_len = len(js['recordings'])
    print(f'{request_len} recordings founds!')
    continue_bool = input("Continue? [y] / [n] / or type the number of files you want to download: ")
    try:
        n_files = int(continue_bool)
        isint = True
    except:
        n_files = request_len
        isint = False
    while (continue_bool not in ['y', 'n']) and (not isint):
        print("Please type y for yes or n for no and press ENTER")
        continue_bool = input("Continue? [y] / [n]: ")
    
    if continue_bool == 'n':
        return []
    
    indexes = np.arange(request_len)
    np.random.shuffle(indexes)
    print('~~Downloading~~')
    n = 0
    for i in tqdm(indexes):
        recording = js['recordings'][i]
        rec_id = recording['id']
        if rec_id in file_ids:
            continue
        filename = recording['gen'].lower() + '_' + recording['sp'].lower() + '#' + recording['id'] + '.mp3'
        urllib.request.urlretrieve(recording['file'], filename=os.path.join(write_path, filename))
        file_ids.append(rec_id)
        n += 1
        if n == n_files:
            break
    print(f'{n} new files added!')
        
    with open(file_ids_path, 'w') as f:
        json.dump(file_ids, f)
    print('Mp3 --> wav...')
    dir_convert_mp32wav(write_path)
    print('Process over!')

    return file_ids


def dir_convert_mp32wav(directory, keep_file=False):
    '''
    Processes a directory, applying file_convert_mp32wav to every mp3 file
    Parameters:
    - str directory: path to directory
    - bool keep_file: whether to delete original mp3 file or not
    '''
    # res = np.array([file_convert_mp32wav(os.path.join(directory, f), keep_file=keep_file) for f in os.listdir(directory) 
    #                 if os.path.splitext(f)[-1] == '.mp3']).sum(axis=0)
    res = np.array([file_convert_mp32wav(mp3_file, keep_file=keep_file) for mp3_file in glob.glob(directory + '/*.mp3')]).sum(axis=0)
    
    print(f'Directory {directory} processed, {res} conversion and deletions')

    
def file_convert_mp32wav(input_file, keep_file=False):
    '''
    Converts a sound file from mp3 to wav using ffmpeg and merges stereo to mono
    Parameters:
    - str input_file: path to file to convert
    - bool keep_file: whether to delete original mp3 file or not
    Returns
    - tuple (int convert, int delete) indicating if conversion/deletion was performed or not
    '''
    
    output_file = '.'.join([os.path.splitext(input_file)[0], 'wav'])
    convert = 0
    delete = 0
    
    if not os.path.isfile(output_file):
        # if output file doesn't already exist
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream, output_file, ac=1)
        ffmpeg.run(stream)
        convert = 1
        
    if not keep_file:
        os.remove(input_file)
        delete = 1
        
    return convert, delete


def download_from_annots(dirp, out_dirp):
    """
    Downloads audio files from a list of txt annotation files ([SPECIES]#[FILE_ID].txt)
    """
    annot_files = glob.glob(dirp + '/*.txt')
    df = pd.DataFrame([os.path.basename(f).replace('.txt', '') for f in annot_files])
    df = df[0].str.split('#', expand=True)
    df = df.groupby(0, as_index=False).agg({1: lambda x: x.tolist()})

    for i in range(len(df)):
        species = df.iloc[i][0]
        file_ids = df.iloc[i][1]
        download_species_ids(species.replace('_', '%20'), file_ids, out_dirp)     


def download_species_ids(species, ids, out_dirp):

    page_number = 0
    required_files = len(ids)
    processed_files = 0

    while processed_files < required_files:
        page_number += 1
        response = requests.get(f'https://xeno-canto.org/api/2/recordings?query={species}&page={str(page_number)}')
        js = response.json()
        if 'error' in js.keys():
            break

        recordings = js['recordings']
        recordings = [e for e in recordings if e['id'] in ids]

        if len(recordings) > 0:
            for j, recording in enumerate(recordings):
                filename = species.replace('%20', '_') + '#' + recording['id']
                if not os.path.exists(os.path.join(out_dirp, filename + '.mp3')):
                    urllib.request.urlretrieve(recording['file'], filename=os.path.join(out_dirp, filename + '.mp3'))
            processed_files += j + 1
            
    bird = species.replace('%20', ' ')
    print(f'{processed_files} files of {bird} done!')

    dir_convert_mp32wav(out_dirp)

    return 1

def main():

    ssl._create_default_https_context = ssl._create_unverified_context # is this necessary?
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', type=str, help='Bird species')
    parser.add_argument('-t', '--sound_type', type=str, help='Which type of sound (call, song, ...)')
    parser.add_argument('-q', '--quality', type=str, help='Quality')
    parser.add_argument('-lt', '--max_length', type=float, help='Max length in seconds')
    parser.add_argument('-o', '--filepath', type=str, help='Where to save audio files')
    args = parser.parse_args()
    download_request(args)

if __name__ == "__main__":
    main()