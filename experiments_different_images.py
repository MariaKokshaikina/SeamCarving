from datetime import datetime

import pandas as pd
import numpy as np

from energy import energy_gradient, energy_function_forward, gradient_magnitude_sobel_operator
from seams import seams_map_dp, seam_map_function_forward, carve_column_mask
from seam_carve import SeamCarve

from PIL import Image

import os

if __name__ == '__main__':

    DATASET_DIR = 'dataset'
    result = []
    for filename in os.listdir(DATASET_DIR):
        img_path = os.path.join(DATASET_DIR, filename)
        img = Image.open(img_path)
        shape = np.array(img).shape
        original_height = shape[0]
        original_width = shape[1]
        for new_height_ratio in [0.75, 1, 1.25]:
            for new_width_ratio in [0.75, 1, 1.25]:
                if new_height_ratio == new_width_ratio == 1:
                    continue
                new_height = round(original_height * new_height_ratio)
                new_width = round(original_width * new_width_ratio)
                new_size = (new_height, new_width)
                for (energy_function, seam_map_function) in [
                    (energy_gradient, seams_map_dp),
                    (energy_function_forward, seam_map_function_forward),
                    (gradient_magnitude_sobel_operator, seams_map_dp),
                ]:
                    seamCarve = SeamCarve(
                        image_path=img_path,
                        new_size=new_size,
                        energy_function=energy_function,
                        seam_map_function=seam_map_function,
                        carve_function=carve_column_mask,
                        importance_map=None,
                        name_suffix=energy_function.__name__)
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
                    print(result[-1])

    result = pd.DataFrame(result)
    result.to_csv('result.csv', index=False, header=True)
