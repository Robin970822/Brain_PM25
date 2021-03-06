from utils import propose_region, bet_unet
from model import load_model, unet
from tqdm import tqdm

import numpy as np
import nibabel as nib

import cv2
import os
import time
import argparse


def detect_file(filename, model, unet):
    data = nib.load(filename)
    mask = nib.load(filename)
    mask_roi = nib.load(filename)

    width, height, frame_num = data.shape
    matrix = data.get_data()
    bet_matrix = np.uint8(bet_unet(matrix, unet))

    start = time.time()
    for i in tqdm(range(frame_num), desc='Detect in {}'.format(os.path.basename(filename))):
        img = matrix[:, :, i]
        bet = bet_matrix[:, :, i]
        selected, mask_ROI = detect(img, model, bet)
        mask.get_data()[:, :, i] = selected
        mask_roi.get_data()[:, :, i] = mask_ROI
    end = time.time()

    print('Using time {}s'.format(end - start))
    return mask, mask_roi


def detect(img, model, bet, pad=10, is_debug=False):
    width, height = img.shape
    mask_ROI = propose_region(img, is_debug) & bet
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
        res = model.predict(slice_img)
        if np.argmax(res) == 1:
            selected[labels == i] = np.max(res)
            if is_debug:
                print('label:{}, center:({},{})'.format(i, cx, cy))
    return selected, mask_ROI


if __name__ == '__main__':
    import config
    data_root = config.data_root
    model_path = config.model_path
    result_path = config.result_path

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input path')
    parser.add_argument('-o', '--output', help='output path')
    parser.add_argument('-m', '--model', help='model path',
                        default='CNN_p10_e2000.h5')
    args = parser.parse_args()

    data_filename = args.input
    data_path = os.path.join(data_root, data_filename)

    result_filename = args.output if args.output else '{}_detect.nii'.format(
        data_filename.split('/')[-1].split('.')[-2])

    result_path = os.path.join(result_path, result_filename)

    model_name = args.model
    unet_path = os.path.join(model_path, 'unet_BET2.hdf5')
    clf_path = os.path.join(model_path, model_name)
    bet_net = unet(pretrained_weights=unet_path)
    model = load_model(clf_path)

    mask, mask_roi = detect_file(data_path, model, bet_net)

    nib.save(mask, result_path)
    nib.save(mask_roi, './mask/roi.nii')
