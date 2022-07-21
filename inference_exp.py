import argparse
import os
import os.path as osp
import subprocess

import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np

def convert_masks(converted_masks, masks):
    reverse_pixel_mapping = {
        0 : 6, ## maps i-th channel non-zero pixels to 
               ## competition prediction space
        1 : 7,
        2 : 10
    }
    # print(masks)
    for file in tqdm(os.listdir(masks)):
        mask = np.array(Image.open(osp.join(masks_path, file)))
        h, w, c = mask.shape
        new_mask = np.zeros((h, w), dtype=mask.dtype)
        for channel, pixel_value in reverse_pixel_mapping.items():
            new_mask = np.where(mask[..., channel] != 0, pixel_value, new_mask)
        new_mask_pil = Image.fromarray(new_mask)
        # print(new_mask_pil.size, '\n', 50 * '*')
        new_mask_pil.save(osp.join(converted_masks, file))

def search_by_prefix(path, prefix):
    try:
        return list(filter(lambda x : x.startswith(prefix), os.listdir(path)))[0]
    except:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='/workspace/rlh/work_dirs')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--opacity', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    exp_dir = args.exp_dir
    exp = args.exp
    opacity = args.opacity

    weights = search_by_prefix(osp.join(exp_dir, exp), 'best')
    if weights is None:
        weights = search_by_prefix(osp.join(exp_dir, exp), 'iter')
        
    checkpoint = osp.join(exp_dir, exp, weights)
    
    sub_num = 1
    while osp.isdir(osp.join(exp_dir, exp, f'sub_{sub_num}')):
        sub_num += 1

    sub_path = osp.join(exp_dir, exp, f'sub_{sub_num}')
    os.makedirs(sub_path)

    masks_path = osp.join(sub_path, f"test_masks_{opacity}")
    config = search_by_prefix(osp.join(exp_dir, exp), 'config')
    from pprint import pprint
    pprint(f"Found config : {config}")
    print("Creating masks using trained model")
    os.makedirs(masks_path)
    subprocess.run(
        f"PYTHONPATH=mmsegmentation \
            python3 mmsegmentation/tools/test.py \
            {osp.join(exp_dir, exp, config)} \
            {checkpoint} \
            --show-dir {masks_path} \
            --opacity {opacity}", shell=True)

    converted_masks_path = osp.join(sub_path, 'converted_masks')
    print("Converting masks to competition format")
    os.makedirs(converted_masks_path)
    convert_masks(converted_masks_path, masks_path)

    print("Creating submission")
    shutil.make_archive(osp.join(sub_path, 'solution'), 'zip', converted_masks_path)
    
    print("Copying config")
    shutil.copyfile(osp.join(exp_dir, exp, config), osp.join(sub_path, config))
    
