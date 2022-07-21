import os
from os.path import join
from PIL import Image

import numpy as np
pixel_mapping = {
    0 : 0,
    6 : 1,
    7 : 2,
    10 : 3
}
if __name__ == '__main__':
    data_dir = '/workspace/datasets/rlh/data/train/mask'
    new_dir_name = '/workspace/datasets/rlh/data/train/new_mask'
    test = False

    os.makedirs(new_dir_name, exist_ok=True)
    for filename in os.listdir(data_dir):
        mask = np.array(Image.open(join(data_dir, filename)))
        mask = mask[..., 0] ## leave single channel out of 3
        for initial_value, new_value in pixel_mapping.items():
            mask = np.where(mask == initial_value, new_value, mask)
        Image.fromarray(mask).save(join(new_dir_name, filename))
        if test:
            break