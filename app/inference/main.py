import os
import json
import glob
from src.models.run_detection_cpu import load_model, run_detection
from src.models.bird_dict import BIRD_DICT
from src.visualization.visu import merge_images, visualise_model_out

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

mod_p = 'models/detr_noneg_100q_bs20_r50dc5'
model, config = load_model(mod_p)

logger.info(BIRD_DICT)

birds_dict = BIRD_DICT.copy()
birds_dict.update({'Non bird sound': 0})
logger.info(birds_dict)
reverse_dict = {id: bird_name for bird_name, id in birds_dict.items()}

# test_dirp = r'../data/external/tests/audio_samples/'
# glob.glob(test_dirp)
# wav_f_p = glob.glob(os.path.join(test_dirp, '*.wav'))
# t_filep = wav_f_p[0]

t_filep = 'inference/Turdus_merlula.wav'

logger.info(f"Starting run_detection on {t_filep.split('/')[-1]}...")
fp, outputs, spectrogram = run_detection(model, config, t_filep, return_spectrogram=True)

logger.info(f"[fp]: \n{fp}\n\n")
logger.info(f"[outputs]: \n{outputs}\n\n")

class_bbox = merge_images(fp, outputs, config.num_classes)
output = {reverse_dict[idx]: {key: value.cpu().numpy().tolist() for key, value in class_bbox[str(idx)].items()} for idx in range(1, len(class_bbox) + 1) if len(class_bbox[str(idx)]['bbox_coord']) > 0}
logger.info(f"[output]: \n{output}")

# visualise_model_out(output, fp, spectrogram, reverse_dict)