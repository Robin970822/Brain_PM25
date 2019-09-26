from utils import crop_from_img, propose_region

import numpy as np
import nibabel as nib

import os
import config
import argparse


def get_train_data(matrix, mask_matrix, frame_num, data_type='pos'):
    data = []
    for i in range(frame_num):
        img = matrix[:, :, i]
        img_mask = mask_matrix[:, :, i]
        if data_type == 'pos':
            img_list = crop_from_img(img, img_mask)
        elif data_type == 'neg':
            img_ROI = propose_region(img, is_debug=False)
            img_neg = img_ROI & (~img_mask)
            img_list = crop_from_img(img, img_neg)
        for img_slice in img_list:
            data.append(img_slice)

    npy_path = os.path.join(result_path, '{}.npy'.format(data_type))
    np.save(npy_path, data)
    print('Write {} {} Data in {}'.format(len(data), data_type, npy_path))
    return data


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--data_type', default='p',
                    help='Type of data to get:\npos for positive\nneg for negative')
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


get_train_data(matrix, mask_matrix, frame_num, data_type=args.data_type)
