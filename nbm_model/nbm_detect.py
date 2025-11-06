import os, glob, argparse
from run_detection import load_model, run_detection



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Bird call detection with NBM model")

    parser.add_argument('--ckpt', dest='model_dirp', type=str, default='model_weights',
                        help='The path to the model weights & cfg directory.')
    parser.add_argument('--audio_dir', dest='audio_dirp', type=str, required=True,
                        help='The path to the directory which contains the wav files to analyze.')
    parser.add_argument('--min_score', type=float, default=0.2,
                        help='Minimum confidence score.')
    parser.add_argument('--batch', dest='bs', type=int, default=4,
                        help='Batch size.')

    args = parser.parse_args()

    assert os.path.isfile('bird_dict.json'), 'Missing dictionary of bird species names --> bird_dict.json.'

    model, model_args = load_model(args.model_dirp)
    for i, wav_path in enumerate(glob.glob(args.audio_dirp + '/*.wav')):
        #### Exec model
        output = run_detection(model, model_args, wav_path, min_score=args.min_score, visualise_outputs=False, bs=args.bs, bird_dicts_path='bird_dict.json')
        with open(wav_path.replace('.wav', '.txt'), 'w') as f:
            f.write(str(output))
        print(f'~~~~~ File {os.path.basename(wav_path).replace('.wav', '')} done ~~~~~')