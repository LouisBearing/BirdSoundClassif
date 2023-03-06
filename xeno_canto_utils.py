import os
import json
import requests
import urllib
import ffmpeg
import numpy as np
import glob
import argparse


def download_request(args):

    # Load list of already processed file indexes

    # with open(os.path.join(r'C:\Users\laeri\NBM\data_2\xc', 'file_ids.json'), 'r') as f:
    #     file_ids = json.load(f)
    # file_ids = file_ids['file_ids']
    file_ids = []
    
    # XC API request    
    parameters = {
        'query': f'{args.species} type:"{args.sound_type}" len_lt: {args.max_length} q:{args.quality}'
    }

    response = requests.get('https://www.xeno-canto.org/api/2/recordings', params=parameters)
    js = response.json()

    print(f'{len(js)} recordings founds!')

    for i in np.arange(len(js['recordings'])):
        recording = js['recordings'][i]
        rec_id = recording['id']
        # if (rec_id in file_ids) or (recording['also'] != ['']) or ('juvenile' in recording['type']):
        #     continue
        # elif (args.sound_type != 'song') and ('song' in recording['type']):
        #     continue
        filename = recording['gen'].lower() + '_' + recording['sp'].lower() + '_' + recording['id'] + '.mp3'
        urllib.request.urlretrieve(recording['file'], filename=os.path.join(args.filepath, filename))
        file_ids.append(rec_id)
        
    # Convert mp3 to wav
    dir_convert_mp32wav(args.filepath, keep_file=False)
        
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', type=str, help='Bird species')
    parser.add_argument('-t', '--sound_type', type=str, help='Which type of sound (call, song, ...)')
    parser.add_argument('-q', '--quality', type=str, help='Quality')
    parser.add_argument('-lt', '--max_length', type=float, help='Max length in seconds')
    parser.add_argument('-o', '--filepath', type=str, help='Where to save audio files')
    args = parser.parse_args()
    download_request(args)