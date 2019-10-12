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


def detect(img, model, pad=config.pad, is_debug=False):
    mask_ROI = propose_region(img, is_debug)
    num, labels, stats, centroid = cv2.connectedComponentsWithStats(
        mask_ROI, connectivity=8)
    selected = np.zeros_like(img)
    for i, stat, center in zip(range(num), stats, centroid):
        if is_debug:
            print(i, stat, center)
        x, y, w, h, area = stat
        # remove background
        if x == 0 and y == 0:
            continue
        cx, cy = np.uint8(center)
        # crop a slice around pickle
        # valid bounder
        if cx - pad < 0 or cx + pad > height or cy - pad < 0 or cy + pad > width:
            continue
        if is_debug:
            print('label:{}, center:({},{})'.format(i, cx, cy))
        slice_img = img[cy - pad: cy + pad, cx - pad: cx + pad]
        slice_img = slice_img[:, :, np.newaxis]
        slice_img = np.expand_dims(slice_img, axis=0)
        if np.argmax(model.predict(slice_img)) == 1:
            selected[labels == i] = 1
            if is_debug:
                print('label:{}, center:({},{})'.format(i, cx, cy))
    return selected, mask_ROI


data_root = config.data_root
model_path = config.model_path
result_path = config.result_path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input path')
parser.add_argument('-o', '--output', help='output path', default='test.nii')
parser.add_argument('-m', '--model', help='model path',
                    default='CNN_h3_e1500.h5')
args = parser.parse_args()

data_filename = args.input
data_path = os.path.join(data_root, data_filename)

result_filename = args.output
result_path = os.path.join(result_path, result_filename)

model_name = args.model
model_path = os.path.join(model_path, model_name)
model = load_model(model_path)

data = nib.load(data_path)
mask = nib.load(data_path)
mask_roi = nib.load(data_path)

width, height, frame_num = data.shape
matrix = data.get_data()

start = time.time()
for i in tqdm(range(frame_num), desc='Detect'):
    img = matrix[:, :, i]
    selected, mask_ROI = detect(img, model)
    mask.get_data()[:, :, i] = selected
    mask_roi.get_data()[:, :, i] = mask_ROI
end = time.time()

print('Using time {}s'.format(end - start))

nib.save(mask, result_path)
nib.save(mask_roi, './mask/roi.nii')
