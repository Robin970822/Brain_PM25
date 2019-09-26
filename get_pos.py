from matplotlib import pyplot as plt
from utils import crop_from_img

import numpy as np
import nibabel as nib

import os
import cv2
import config

data_root = config.data_root

data_filename = '0710_60day_20190911_110457SWI.nii'
mask_filename = '07-110.nii'

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

pos = []
for i in range(frame_num):
    img = matrix[:, :, i]
    img_mask = mask_matrix[:, :, i]
    img_list = crop_from_img(img, img_mask)
    for j, img_slice in enumerate(img_list):
        # img_name = os.path.join(result_path, 'pos_{}_{}.png'.format(i, j))
        plt.imshow(img_slice, cmap='bone')
        # cv2.imwrite(img_name, img_slice)
        # plt.imsave(img_name, img_slice)
        pos.append(img_slice)

npy_path = os.path.join(result_path, 'pos.npy')
np.save(npy_path, pos)
