import cv2
import numpy as np

from matplotlib import pyplot as plt


# crop slice from img according to connected components in mask
def crop_from_img(img, mask, pad=5, is_debug=False):
    width, height = img.shape
    img_list = []
    mask = np.uint8(mask)
    num, labels, stats, centroid = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    for i, stat, center in zip(range(num), stats, centroid):
        if is_debug:
            print(i, stat, center)
        x, y, w, h, area = stat
        # remove background
        if x == 0 and y == 0:
            continue
        cx, cy = np.uint8(center)
        # crop a slice raound pickle
        # valid bounder
        if cx - pad < 0 or cx + pad > height or cy - pad < 0 or cy + pad > width:
            print('cross bounder, label:{}, center:({},{})'.format(i, cx, cy))
            continue
        if is_debug:
            print('label:{}, center:({},{})'.format(i, cx, cy))
        slice_img = img[cy - pad: cy + pad, cx - pad: cx + pad]
        if is_debug:
            plt.imshow(slice_img, cmap='bone')
        img_list.append(slice_img)
    return img_list


def generate_from_file_list(file_list):
    data = np.concatenate(tuple([np.load(f) for f in file_list]), axis=0)
    data = data[:, :, :, np.newaxis]
    return data
