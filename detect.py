from utils import propose_region
from model import load_model
from tqdm import tqdm

import numpy as np
import nibabel as nib

import cv2
import os
import time
import config
import argparse

data_root = config.data_root
is_debug = False
data_filename = '0710_60day_20190911_143703SWI.nii'
mask_filename = '07-143m.nii'
data_path = os.path.join(data_root, data_filename)
mask_path = os.path.join(data_root, mask_filename)

model_name = 'CNN_h3_e1000.h5'
model_root = os.path.join(data_root, 'model')
model_path = os.path.join(model_root, model_name)
model = load_model(model_path)

data = nib.load(data_path)
mask = nib.load(mask_path)
mask_roi = nib.load(mask_path)

assert data.shape == mask.shape
width, height, frame_num = data.shape
matrix = data.get_data()
mask_matrix = mask.get_data()

start = time.time()
for i in tqdm(range(frame_num), desc='Detect'):
    img = matrix[:, :, i]
    mask_ROI = propose_region(img, is_debug)
    num, labels, stats, centroid = cv2.connectedComponentsWithStats(mask_ROI, connectivity=8)
    pad = 5
    slice_list = []
    selected = np.zeros_like(img)
    selected_dict = {}
    for ii, stat, center in zip(range(num), stats, centroid):
        if is_debug:
            print(ii, stat, center)
        x, y, w, h, area = stat
        # remove background
        if x == 0 and y == 0:
            continue
        cx, cy = np.uint8(center)
        # crop a slice raound pickle
        # valid bounder
        if cx - pad < 0 or cx + pad > height or cy - pad < 0 or cy + pad > width:
            # print('cross bounder, label:{}, center:({},{})'.format(ii, cx, cy))
            continue
        if is_debug:
            print('label:{}, center:({},{})'.format(i, cx, cy))
        slice_img = img[cy - pad: cy + pad, cx - pad: cx + pad]
        slice_img = slice_img[:, :, np.newaxis]
        slice_img = np.expand_dims(slice_img, axis=0)
        if np.argmax(model.predict(slice_img)) == 1:
            selected_dict.update({ii: {'x': cx, 'y': cy, 'img': slice_img}})
            selected[labels == ii] = 1
            if is_debug: print('label:{}_{}, center:({},{})'.format(i, ii, cx, cy))
    mask.get_data()[:, :, i] = selected
    mask_roi.get_data()[:, :, i] = mask_ROI
end = time.time()

print('Using time {}s'.format(end-start))

nib.save(mask, './test_cnn.nii')
nib.save(mask_roi, './roi.nii')