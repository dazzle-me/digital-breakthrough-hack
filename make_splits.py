import pandas as pd
import numpy as np
from PIL import Image
import tqdm

import warnings
warnings.filterwarnings("ignore")
from skmultilearn.model_selection import IterativeStratification
import os
from os.path import join

if __name__ == '__main__':
    data_root = '/workspace/data'
    base_dir = '/workspace/data/train'
    mask_dir = join(base_dir, 'new_mask')
    df = pd.DataFrame(columns=['id', 'background', 'main_rails', 'side_rails', 'train'])
    test = False
    if not os.path.isfile(join(base_dir, 'meta.csv')):
        for filename in tqdm.tqdm(os.listdir(mask_dir)):
            mask = Image.open(join(mask_dir, filename))
            values = np.unique(mask)
            value_dict = {
                "id" : filename.replace('.png', '')
            }
            for i in range(4):
                for mask_value, column in enumerate(['background', 'main_rails', 'side_rails', 'train']):
                    value_dict[column] = mask_value in values
            df = df.append(
                value_dict,
                ignore_index=True
            )  
            if test:
                print(values)
                break
        df.to_csv(join(base_dir, 'meta.csv'), index=False)
    else:
        df = pd.read_csv(join(base_dir, 'meta.csv'))
    k_fold = IterativeStratification(n_splits=15, order=1)
    label_columns = ['background', 'main_rails', 'side_rails', 'train']    
    
    split_dir = join(data_root, 'stratified_split')
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
    for train_indices, val_indices in zip([range(len(df))], [[]]):
        print(f"Num train samples : {len(train_indices)}, num val samples : {len(val_indices)}")
        
        for column in label_columns:
            print('*' * 20)
            print(f"Class : {column}")
            print(df.loc[train_indices][column].value_counts())
            print(df.loc[val_indices][column].value_counts())
            print('*' * 20)
        with open(join(split_dir, 'train.txt'), 'w') as file:
            for label in df['id'][train_indices]:
                file.write(label + '\n')
        with open(join(split_dir, 'val.txt'), 'w') as file:
            for label in df['id'][val_indices]:
                file.write(label + '\n')
        break

    
    
