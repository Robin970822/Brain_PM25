from model import load_model
from tqdm import tqdm

import numpy as np
import nibabel as nib

import cv2
import os
import time
import argparse


def detect_file(filename, model, distant_map, pad=10, is_debug=False):
    data = nib.load(filename)
    mask = nib.load(filename)
    mask_roi = nib.load(filename)

    width, height, frame_num = data.shape
    matrix = data.get_data()

    start = time.time()
    for i in tqdm(range(frame_num), desc='Detect in {}'.format(os.path.basename(filename))):
        img = matrix[:, :, i]
        dm = distant_map[:, :, i]
        selected, mask_ROI = detect(img, model, dm)
        mask.get_data()[:, :, i] = selected
        mask_roi.get_data()[:, :, i] = mask_ROI
    end = time.time()

    print('Using time {}s'.format(end - start))
    return mask, mask_roi


def detect(img, model, dm, pad=10, is_debug=False):
    width, height = img.shape

    thres = 0.95
    mask_ROI = np.zeros_like(dm)
    mask_ROI[dm > thres] = 1
    mask_ROI = np.uint8(mask_ROI)

    # mask_ROI = propose_region(img, is_debug)
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


if __name__ == '__main__':
    import config
    data_root = config.data_root
    model_path = config.model_path
    result_path = config.result_path

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input path',
                        default='0710_60day_20190911_110457SWI.nii')
    parser.add_argument('-o', '--output', help='output path')
    parser.add_argument('-d', '--distant', help='distant map path',
                        default='07-110_dis.npy')
    parser.add_argument('-m', '--model', help='model path',
                        default='CNN_p10_e2000.h5')
    args = parser.parse_args()

    data_filename = args.input
    data_path = os.path.join(data_root, data_filename)

    distant_filename = args.distant
    distant_path = os.path.join(data_root, distant_filename)
    distant_map = np.load(distant_path)

    result_filename = args.output if args.output else '{}_detect_test_dm.nii'.format(
        data_filename.split('/')[-1].split('.')[-2])

    result_path = os.path.join(result_path, result_filename)

    model_name = args.model
    model_path = os.path.join(model_path, model_name)
    model = load_model(model_path)

    mask, mask_roi = detect_file(data_path, model, distant_map)

    nib.save(mask, result_path)
    nib.save(mask_roi, './mask/roi_test_dm.nii')
