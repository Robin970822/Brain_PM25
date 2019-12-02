import cv2
import numpy as np
from matplotlib import pyplot as plt


# min max scaler
def min_max_scaler(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))


# crop roi for unet prediction
def crop_roi(img):
    test_img = np.rot90(min_max_scaler(img))
    crop_img = test_img[130:130 + 160, 50:50 + 224]
    return crop_img


# resize crop roi to full image
def resize_img(crop, img_shape=(300, 300)):
    mask = np.zeros(img_shape)
    mask[130:130 + 160, 50:50 + 224] = crop
    return np.rot90(mask, 3)


# crop slice from img according to connected components in mask
def crop_from_img(img, mask, pad, is_debug=False):
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
        # crop a slice around pickle
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
def propose_region(img, unet=None, is_debug=False):
    img = np.uint8(img)
    mask = np.zeros_like(img)
    # BET
    if unet:
        bet = unet_predict(img, unet, threshold=0.2)
        bet = np.uint8(bet)

    # Adaptive Threshold
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

    # VH img
    v_img = threshold.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    v_img = cv2.morphologyEx(v_img, cv2.MORPH_OPEN, kernel)

    h_img = threshold.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    h_img = cv2.morphologyEx(h_img, cv2.MORPH_OPEN, kernel)

    vh_img = v_img | h_img
    if is_debug:
        plt.imshow(vh_img, cmap='bone')

    if unet:
        return threshold & bet & mask & (~vh_img)
    else:
        return threshold & mask & (~vh_img)


# generate dataset from file list
def generate_from_file_list(file_list):
    data = np.concatenate(tuple([np.load(f) for f in file_list]), axis=0)
    data = data[:, :, :, np.newaxis]
    return data


# random shuffle dataset
def random_shuffle(data, label):
    assert data.shape[0] == label.shape[0]
    sample_length = data.shape[0]
    index = list(range(sample_length))
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


# dataset augmentation
def augmentation(data, threshold=0.8):
    sample_length = data.shape[0]
    for i in range(sample_length):
        # flip
        if np.random.random() > threshold:
            img = data[i].copy()
            data[i] = np.reshape(
                cv2.flip(img, flipCode=np.random.randint(-1, 1)), img.shape)
    return data


# unet predict
def unet_predict(img, model, threshold=0.5):
    # crop img
    h, w = img.shape
    crop_img = crop_roi(img)
    crop_img = np.reshape(crop_img, crop_img.shape + (1,))
    crop_img = np.reshape(crop_img, (1,) + crop_img.shape)

    # predict res
    res = model.predict(crop_img)
    res = np.squeeze(res)
    r = np.zeros_like(res)
    r[res > threshold] = 1

    # mask
    return resize_img(r, (w, h))


# unet batch predict
def unet_batch_predict(img_batch, model, threshold=0.5):
    # img_batch: [batch w, h]
    batch, h, w = img_batch.shape
    crop_batch = []
    for img in img_batch:
        crop_img = crop_roi(img)
        crop_batch.append(crop_img)
    crop_batch = np.array(crop_batch)
    crop_batch = np.reshape(crop_batch, crop_batch.shape + (1,))

    # predict res
    res = model.predict_on_batch(crop_batch)
    res = np.squeeze(res)
    r_batch = np.zeros_like(res)
    r_batch[res > threshold] = 1

    # mask_batch: [batch, w, h]
    mask_batch = [resize_img(r, (w, h)) for r in r_batch]
    return mask_batch
