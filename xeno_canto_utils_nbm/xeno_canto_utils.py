import os
import json
import requests
import urllib
# import ffmpeg
import numpy as np
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
    for i in tqdm(indexes[:n_files]):
        recording = js['recordings'][i]
        rec_id = recording['id']
        if rec_id in file_ids:
            continue
        filename = recording['gen'].lower() + '_' + recording['sp'].lower() + '#' + recording['id'] + '.mp3'
        urllib.request.urlretrieve(recording['file'], filename=os.path.join(write_path, filename))
        file_ids.append(rec_id)
        
    with open(file_ids_path, 'w') as f:
        json.dump(file_ids, f)
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
    Converts a sound file from mp3 to wav using ffmpeg
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
        stream = ffmpeg.output(stream, output_file)
        ffmpeg.run(stream)
        convert = 1
        
    if not keep_file:
        os.remove(input_file)
        delete = 1
        
    return convert, delete

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