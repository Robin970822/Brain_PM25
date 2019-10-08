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

# propose region from image


def propose_region(img, is_debug):
    img = np.uint8(img)
    mask = np.zeros_like(img)
    threshold = cv2.adaptiveThreshold(
        img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 3)
    if is_debug:
        plt.imshow(threshold, cmap='bone')

    num, labels, stats, centroid = cv2.connectedComponentsWithStats(
        threshold, connectivity=8)
    for i in range(num):
        x, y, w, h, area = stats[i, :]
        if area > 100 or (w / (h + 1e-4)) > 5 or (h / (w + 1e-4)) > 5:
            continue
        mask[labels == i] = 1
    if is_debug:
        plt.imshow(mask, cmap='bone')

    v_img = threshold.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    v_img = cv2.morphologyEx(v_img, cv2.MORPH_OPEN, kernel)

    h_img = threshold.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_img = cv2.morphologyEx(h_img, cv2.MORPH_OPEN, kernel)

    vh_img = v_img | h_img
    if is_debug:
        plt.imshow(vh_img, cmap='bone')

    return threshold & mask & (~vh_img)


def generate_from_file_list(file_list):
    data = np.concatenate(tuple([np.load(f) for f in file_list]), axis=0)
    data = data[:, :, :, np.newaxis]
    return data
