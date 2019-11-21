from distant import get_seeds, compute_dm_rasterscan, compute_mp
from tqdm import tqdm

import numpy as np
import nibabel as nib

import os
import config
import argparse


def get_train_data(matrix, mask_matrix, frame_num, result_path):
    data = np.zeros_like(matrix)
    for i in tqdm(range(frame_num), desc=result_path):
        img = matrix[:, :, i]
        img_mask = mask_matrix[:, :, i]
        seeds = get_seeds(img_mask)
        dist_map = compute_dm_rasterscan(
            img, seeds, its=2, dist_type='geodesic')
        mp = compute_mp(dist_map, p=5)
        data[:, :, i] = mp

    npy_path = os.path.join(result_path, 'distant_map.npy')
    np.save(npy_path, data)
    print('Write {} Data in {}'.format(len(data), npy_path))
    return data


parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--data', default='0710_60day_20190911_110457SWI.nii', help='input file')
parser.add_argument('-m', '--mask', default='07-110.nii',
                    help='input mask file')
args = parser.parse_args()

# Data Path
data_root = config.data_root

data_filename = args.data
mask_filename = args.mask

data_path = os.path.join(data_root, data_filename)
mask_path = os.path.join(data_root, mask_filename)
result_path = os.path.join(data_root, data_filename.split('.')[0])
if not os.path.exists(result_path):
    os.mkdir(result_path)

data = nib.load(data_path)
mask = nib.load(mask_path)

assert data.shape == mask.shape
width, height, frame_num = data.shape

matrix = data.get_data()
mask_matrix = mask.get_data()

get_train_data(matrix, mask_matrix, frame_num, result_path)
