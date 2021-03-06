from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from energy import energy_gradient, energy_function_forward, gradient_magnitude_sobel_operator
from seams import seams_map_dp, seam_map_function_forward, carve_column_mask
from seam_carve import SeamCarve

from PIL import Image

import os

from utils import get_config

config = get_config()

if __name__ == '__main__':

    # MAX_SIZE = 500
    #
    # if not os.path.exists('dataset_500'):
    #     os.mkdir('dataset_500')
    #
    # for filename in os.listdir('dataset'):
    #     img = Image.open(os.path.join('dataset', filename))
    #     w, h = img._size
    #     coef = MAX_SIZE / max(w, h)
    #     new_size = w * coef, h * coef
    #     img.thumbnail(new_size)
    #     img.save(os.path.join('dataset_500', filename))

    DATASET_DIR = config['dataset_dir']
    if not os.path.exists(DATASET_DIR):
        print('Dataset dir doesn\'t exist, exit(0)')
        exit(0)

    STATS_DIR = config['stats_dir']
    if not os.path.exists(STATS_DIR):
        os.mkdir(STATS_DIR)

    result_filename = os.path.join(STATS_DIR, f'result_{datetime.now()}.csv')
    result = []

    experiments = []
    for filename in sorted(os.listdir(DATASET_DIR)):
        for new_height_ratio in [0.75, 1, 1.25]:
            for new_width_ratio in [0.75, 1, 1.25]:
                if new_height_ratio == new_width_ratio == 1:
                    continue
                for (energy_function, seam_map_function) in [
                    (energy_gradient, seams_map_dp),
                    (energy_function_forward, seam_map_function_forward),
                    (gradient_magnitude_sobel_operator, seams_map_dp),
                ]:
                    experiments.append({
                        'filename': filename,
                        'new_height_ratio': new_height_ratio,
                        'new_width_ratio': new_width_ratio,
                        'energy_function': energy_function,
                        'seam_map_function': seam_map_function,
                    })

    try:
        for experiment in tqdm(experiments):
            filename = experiment['filename']
            new_height_ratio = experiment['new_height_ratio']
            new_width_ratio = experiment['new_width_ratio']
            energy_function = experiment['energy_function']
            seam_map_function = experiment['seam_map_function']

            img_path = os.path.join(DATASET_DIR, filename)
            img = Image.open(img_path)
            shape = np.array(img).shape
            original_height = shape[0]
            original_width = shape[1]
            new_height = round(original_height * new_height_ratio)
            new_width = round(original_width * new_width_ratio)
            new_size = (new_height, new_width)

            seamCarve = SeamCarve(
                image_path=img_path,
                new_size=new_size,
                energy_function=energy_function,
                seam_map_function=seam_map_function,
                carve_function=carve_column_mask,
                importance_map=None,
                name_suffix=energy_function.__name__,
                # save_gif=True,
            )

            if os.path.exists(seamCarve.get_new_img_filename()):
                continue

            time_start = datetime.now()
            seamCarve.run()
            time_end = datetime.now()
            new_img_filename, _ = seamCarve.save_result()
            result.append({
                'original_filename': img_path,
                'original_height': original_height,
                'original_width': original_width,
                'energy_function': energy_function.__name__,
                'new_height': new_height,
                'new_width': new_width,
                'new_filename': new_img_filename,
                'time': (time_end - time_start).microseconds,
            })

    except Exception as e:
        result = pd.DataFrame(result)
        result.to_csv(result_filename, index=False, header=True)
        raise e

    finally:
        result = pd.DataFrame(result)
        result.to_csv(result_filename, index=False, header=True)
