from utils import bet_unet
from model import unet
from tqdm import tqdm
import nibabel as nib

import os
import time
import argparse


def detect_file(filename, model, unet):
    data = nib.load(filename)
    mask = nib.load(filename)
    mask_roi = nib.load(filename)

    width, height, frame_num = data.shape
    matrix = data.get_data()
    unet_matrix = bet_unet(matrix, unet, threshold=0.2)

    start = time.time()
    for i in tqdm(range(frame_num), desc='Detect in {}'.format(os.path.basename(filename))):
        bet = unet_matrix[:, :, i]
        mask.get_data()[:, :, i] = bet
        mask_roi.get_data()[:, :, i] = bet
    end = time.time()

    print('Using time {}s'.format(end - start))
    return mask, mask_roi


if __name__ == '__main__':
    import config
    data_root = config.data_root
    model_path = config.model_path
    result_path = config.result_path

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input path')
    parser.add_argument('-o', '--output', help='output path')
    args = parser.parse_args()

    data_filename = args.input
    data_path = os.path.join(data_root, data_filename)

    result_filename = args.output if args.output else '{}_detect_by_unet.nii'.format(
        data_filename.split('/')[-1].split('.')[-2])

    result_path = os.path.join(result_path, result_filename)

    unet_path = os.path.join(model_path, 'unet_pm25_yuzq.hdf5')
    bet_net = unet(pretrained_weights=unet_path)

    mask, mask_roi = detect_file(data_path, None, bet_net)

    nib.save(mask, result_path)
    nib.save(mask_roi, './mask/roi_by_unet.nii')
