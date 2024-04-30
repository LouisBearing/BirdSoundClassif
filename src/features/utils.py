import os
import ffmpeg
import numpy as np
import pandas as pd
import json
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import glob
import imageio
import matplotlib.pyplot as plt

#####
# Requires the above packages, if not installed: pip install <package name>
#####



def dir_convert_mp32wav(directory, keep_file=False):
    '''
    Processes a directory, applying file_convert_mp32wav to every mp3 file
    Parameters:
    - str directory: path to directory
    - bool keep_file: whether to delete original mp3 file or not
    '''
    res = np.array([file_convert_mp32wav(os.path.join(directory, f), keep_file=keep_file) for f in os.listdir(directory) 
                    if os.path.splitext(f)[-1] == '.mp3']).sum(axis=0)
    
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


def read_txt_file(file, extra_str_label=''):
    '''
    Opens a text file and copy content in a pandas df. Audacity outputs that span 2 lines are put back to a single line
    Parameters:
    - str file: path to txt file
    - str extra_str_label: the substring that is added in label files, depending of the birder
    Returns 
    dataframe containing txt file information
    '''
    
    df = pd.read_table(file, header=None)

    # Separate time and frequency lines (the latter start with "\")
    df['line_type'] = (df[0] == '\\').astype(int)

    # Assign recording id to each line
    df['id'] = [elt for elt in assign_idx(df['line_type'])]
    
    # Suppression of duplicate entries
    df.drop_duplicates(['line_type', 'id'], inplace=True)
    
    # From two to one row per recording
    df = df.loc[df['line_type'] == 0].merge(df.loc[df['line_type'] == 1], on='id').dropna().rename(columns={
    '0_x': 't_start', '1_x': 't_end', '2_x': 'species', '1_y': 'f_start', '2_y': 'f_end'
    })
    df = df[['t_start', 't_end', 'f_start', 'f_end', 'species']]

    df['filename'] = os.path.basename(file).split('.')[0]
    df['filename'] = df['filename'].str.replace(extra_str_label, '')

    for dt_col in ['t_start', 't_end']:
        df[dt_col] = df[dt_col].astype(float)
    
    return df


def create_label_dataset(directory, extra_str_label='', suppress_others=True, suppress_noise=True, suppress_unID=False, is_csv=False):
    '''
    Concatenates text files contents into a pandas df containing recordings information
    Parameters:
    - str directory: path to directory
    - str extra_str_label: the substring that is added in label files, depending of the birder
    - bool suppress_others: whether or not to suppress all non-bird signal
    - bool suppress_noise: whether or not to suppress all background sound or noise
    - bool suppress_others: whether or not to suppress unidentified bird sounds
    Returns:
    dataframe containing recordings information
    '''

    # Bird species to ID dictionary
    dict_dir = r''
    with open(os.path.join(dict_dir, 'bird_dict.json'), 'r') as f:
        birds_dict = json.load(f)
    
    annot_dir = os.path.join(directory, "annotations")
    # Labels dataframe
    if is_csv:
        labels = pd.read_csv(os.path.join(annot_dir, 'annotations.csv'))
        # Suppress file extension
        labels['filename'] = labels['filename'].str.slice_replace(-4, repl='')
    else:
        df_list = [read_txt_file(os.path.join(annot_dir, f), extra_str_label=extra_str_label) 
            for f in os.listdir(annot_dir) if os.path.splitext(f)[-1] == '.txt']
        labels = pd.concat(df_list)

    # Convert to float and clip
    for freq in ['f_start', 'f_end']:
        labels[freq] = labels[freq].astype(float)
    labels['f_start'] = labels['f_start'].clip(lower=0)
    labels.loc[labels['f_end'] < 0, 'f_end'] = 20000    
    
    # Deduplication of recording labels, label with the largest frequency range is kept
    labels['f_delta'] = labels['f_end'] - labels['f_start']
    labels = labels.sort_values('f_delta', ascending=False).drop_duplicates(['filename', 't_start', 'species']).sort_values(['filename', 't_start'])
    del labels['f_delta']
    
    # Clean species
    labels['species'] = labels['species'].map(lambda x: replacements[x] if x in replacements.keys() else x)

    # Assign species to id
    labels['bird_id'] = labels['species'].map(lambda x: birds_dict[x] if x in birds_dict.keys() else np.nan)

    # Other sounds - Attention Background et dérivés (p-ê vent aussi) sont à traiter séparément et ne doivent pas être détectés -> on peut
    # les garder comme samples négatifs (donc label = -1 dans le RPN) contrairement aux autres qui seront plutôt détectés mais classifiés comme
    # "autres sons" par le RCNN.
    noise_labels = ['Bruit de fond', 'Background', 'Backgroud', 'passage moto au loin', 'Back ground', 'Back groung', 'Backgroun', 'Bakground', 'backgroound',
    'background', 'bruit de fond']
    labels.loc[labels['species'].isin(noise_labels), 'bird_id'] = -1

    not_bird_labels = ['Capreolus capreolus', 'Pelophylax sp.', 'Vulpes vulpes', 'Oecanthus pellucens', 'ruspolia nitidula', 'orthoptère', 'voix humaine', 
    'saturation HF par orthoptères', 'Cervus elaphus brame', 'Sus scrofa', 'chien', 'Hannetons par milliers', 'possible battement d\'aile', 'What ??',
    'parasite', 'bruit parasite', 'geophonie', 'Vent geophonie', 'vulpes vulpes', 'Capreolus capreolus ', '0: Bruit parasite', '0: Other biophonia', 
    '0: Other antropophonia', '0: Other geophonia', '0: Background', '1: Autre biophonie', '1: Autre antropophonie', '0: Unknown', 'Inconnu'] # 
    mask_others = labels['species'].map(lambda x: 'autre' in x.lower()) 
    labels.loc[mask_others | labels['species'].isin(not_bird_labels), 'bird_id'] = 0

    # All rarer and unidentified birds (supprimer oiseaux sp. du ds d'apprentissage pour la classif ?)
    # max_idx = len(birds_dict)
    # labels['bird_id'].fillna(max_idx + 1, inplace=True)
    labels['bird_id'].fillna(birds_dict['Other'], inplace=True)
    labels['bird_id'] = labels['bird_id'].astype(int)

    if suppress_noise:
        labels = labels.loc[labels['bird_id'] != -1]
    
    if suppress_others:
        labels = labels.loc[labels['bird_id'] != 0]

    if suppress_unID:
        labels = labels.loc[~labels['species'].isin(['Oiseau sp', 'Parus sp'])]

    labels.index = range(len(labels))
    
    return labels


def assign_idx(col):
    '''
    Function to be used in read_txt_file, increment a recording index each time a new recording is detected, 
    corresponding to a float value in column 0
    '''
    
    idx = -1
    
    for elt in col:
        if elt == 0:
            idx += 1
        yield idx


replacements = {
    'Emberiza ortulana': 'Emberiza hortulana',
    'bernicla bernicla': 'Branta bernicla',
    'Bernicla bernicla': 'Branta bernicla',
    'Grus grus adulte': 'Grus grus',
    'Corvus corone alarme': 'Corvus corone',
    'Phasianus colchicus ': 'Phasianus colchicus',
    'Luscinia megarynchos megarynchos': 'Luscinia megarhynchos',
    'Luscinia megarynchos megarynchos': 'Luscinia megarhynchos',
    'Luscinia megarhynchos megarhynchos ': 'Luscinia megarhynchos',
    'Luscinia megarhynchos megarhynchos': 'Luscinia megarhynchos',
    'Grus grus juvénile': 'Grus grus',
    'Strix aluco chant': 'Strix aluco',
    'Strix aluco cris': 'Strix cris',
    'tachybaptus ruficollis': 'Tachybaptus ruficollis',
    'Tachybaptus ruficollis ': 'Tachybaptus ruficollis',
    'Burhinus burhinus': 'Burhinus oedicnemus',
    'Erithacus rubecula ': 'Erithacus rubecula',
    'Turdus merula alarme': 'Turdus merula',
    'Luscinia megarhynchos': 'Luscinia megarhynchos',
    'Burhinus oedicnemus ' : 'Burhinus oedicnemus',
    'Gallinula chloropus ': 'Gallinula chloropus',
    'chant Luscinia megarhynchos': 'Luscinia megarhynchos',
    'Anas platychyncos': 'Anas platyrhynchos',
    'Grus grus cris': 'Grus grus',
    'Turdus merula cris': 'Turdus merula',
    'Turdus philomelos cris': 'Turdus philomelos',
    'Turdus iliacus cris': 'Turdus iliacus',
    'Erithacus rubecola': 'Erithacus rubecula',
    'Anas platyrhynchos ': 'Anas platyrhynchos',
    'Certhia brachydactyla ': 'Certhia brachydactyla',
    'Streptopelia decaocto ': 'Streptopelia decaocto',
    'Strix aluco ': 'Strix aluco',
    'Botaurus stellaris ': 'Botaurus stellaris',
    'Numenius arquata XC570503': 'Numenius arquata',
    'Chevalier sylvain': 'Tringa glareola',
    'caprimulgus europaeus': 'Caprimulgus europaeus',
    'ardea cinerea': 'Ardea cinerea',
    'Cuculus canorus canorus': 'Cuculus canorus',
    'Charadrius dubius curonicus': 'Charadrius dubius',
    'Charadrius curonicus': 'Charadrius dubius',
    'Erithacus rubecula rubecula': 'Erithacus rubecula',
    'Tyto alba alba': 'Tyto alba',
    'Ardea nycticorax': 'Nycticorax nycticorax',
    'Carduelis carduelis ': 'Carduelis carduelis'
}


def visualise_file_annot(dirp, max_show=None):
    
    annotp = os.path.join(dirp, 'annotations.csv')
    if os.path.exists(annotp):
        annot = pd.read_csv(annotp, sep=';')
        annot['coord'] = annot['coord'].apply(eval)
        annot['bird_id'] = annot['bird_id'].apply(eval)
    else:
        annot = pd.DataFrame({key: [] for key in ['index', 'coord', 'bird_id']})
    filesp = glob.glob(dirp + '/*.png')
    
    # Bird dict
    dict_dir = r''
    with open(os.path.join(dict_dir, 'bird_dict.json'), 'r') as f:
        birds_dict = json.load(f)

    birds_dict.update({'Non bird sound': 0})
    reverse_dict = {id: bird_name for bird_name, id in birds_dict.items()}

    if (max_show is not None) and len(filesp) > max_show:
        show_paths = np.random.choice(filesp, max_show, replace=False)
    else:
        show_paths = filesp
    
    for filep in filesp:

        if filep not in show_paths:
            continue
        
        fileidx = int(os.path.basename(filep).replace('.png', '').split('__')[-1])
        
        file = imageio.imread(filep)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(file, origin='lower')

        annot_img = annot.loc[annot['index'] == fileidx]
        if len(annot_img) > 0:
            idx  = annot_img.index[0]
            bboxes = annot_img['coord'][idx]
            bird_ids = annot_img['bird_id'][idx]

            for j, bbox in enumerate(bboxes):

                x_1, y_1, x_2, y_2 = bbox

                # Create a Rectangle patch
                rect = patches.Rectangle((x_1, y_1), x_2 - x_1, y_2 - y_1, linewidth=1, edgecolor='b', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                y_anchor = y_1 - 20
                if y_anchor < 10:
                    y_anchor = y_2 + 15
                species = reverse_dict[bird_ids[j]]

                # Add the patch to the Axes
                ax.add_patch(rect)
                ax.annotate(f'{species}', (x_1, y_anchor), backgroundcolor='b', color='white', fontsize='small')
                pix_precision_y = 33.3
                pix_precision_x = 0.002993197278911565 # 0.003
                y_labels = [500 + int(y * pix_precision_y) for y in ax.get_yticks()]
                x_labels = [int(1000 * (x + fileidx * 819) * pix_precision_x) / 1000 for x in ax.get_xticks()]
                ax.yaxis.set_major_locator(mticker.FixedLocator(ax.get_yticks().tolist()))
                ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks().tolist()))
                ax.set_xticklabels(x_labels)
                ax.set_yticklabels(y_labels)
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [s]')
        plt.show()