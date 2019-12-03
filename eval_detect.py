import config
import os
import cv2
import argparse

import numpy as np
import pandas as pd
import nibabel as nib

from matplotlib import pyplot as plt


# calculate iou
def iou(gt, pred, threshold=0):
    gt = np.uint8(gt)
    p = np.zeros_like(pred)
    p[pred > threshold] = 1
    p = np.uint8(p)
    i = np.sum(gt & p)
    u = np.sum(gt | p)
    return i / (u + 1e-4)


# calculate ap
def ap(gt, pred, threshold=0):
    gt = np.uint8(gt)
    p = np.zeros_like(pred)
    p[pred > threshold] = 1
    p = np.uint8(p)
    gt_num, gt_labels, gt_stats, gt_centroid = cv2.connectedComponentsWithStats(
        gt, connectivity=8)
    p_num, p_labels, p_stats, p_centroid = cv2.connectedComponentsWithStats(
        p, connectivity=8)

    tp = 0
    for gt_label in range(gt_num):
        gt_cc = np.zeros_like(gt)
        gt_cc[gt_labels == gt_label] = 1
        gt_cc = np.uint8(gt_cc)

        for p_label in range(p_num):
            p_cc = np.zeros_like(p)
            p_cc[p_labels == p_label] = 1
            p_cc = np.uint8(p_cc)
            if iou(gt_cc, p_cc) > 0.3:
                tp = tp + 1
    fp = p_num - tp
    precision = tp / p_num
    recall = tp / gt_num
    return precision, recall, tp, fp, gt_num, p_num


# calculate map
def mAP(gt_matrix, pred_matrix, frame_num, threshold=0.1):
    tp = 0
    fp = 0
    gt_num = 0
    p_num = 0
    for i in range(frame_num):
        _, _, _tp, _fp, _gt_num, _p_num = ap(
            gt_matrix[:, :, i], pred_matrix[:, :, i], threshold)
        tp += _tp
        fp += _fp
        gt_num += _gt_num
        p_num += _p_num
    precision = tp / p_num
    recall = tp / gt_num
    return precision, recall, tp, fp, gt_num, p_num


if __name__ == '__main__':
    data_root = config.data_root
    result_path = config.result_path

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--predict', help='predict for detection', default='0710_094_by_bet.nii')
    parser.add_argument('-g', '--groundtruth',
                        help='ground truth for detection', default='07-094.nii')
    parser.add_argument('-m', '--method', help='detection method', default='U-net BET & CNN classifer')
    args = parser.parse_args()

    method = args.method
    # predict and ground truth path
    predict_path = args.predict
    predict_path = os.path.join(result_path, predict_path)

    groundtruth_path = args.groundtruth
    groundtruth_path = os.path.join(data_root, groundtruth_path)

    # read data from path
    predict = nib.load(predict_path)
    groundtruth = nib.load(groundtruth_path)
    assert groundtruth.shape == predict.shape
    width, height, frame_num = groundtruth.shape
    pred_matrix = predict.get_data()
    gt_matrix = groundtruth.get_data()

    thres = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print('Calculating IOU...')
    ious_imgwise = [[iou(gt_matrix[:, :, i], pred_matrix[:, :, i], thr)
                     for i in range(frame_num)] for thr in thres]
    ious_brainwise = [iou(gt_matrix, pred_matrix, thr) for thr in thres]

    df = pd.DataFrame(np.array(ious_imgwise).T, columns=thres)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                 medians='DarkBlue', caps='Gray')
    df.plot.box(grid=True, color=color,
                title='IOU @threshold with {}'.format(method))
    plt.xlabel('Threshold')
    plt.ylabel('IOU')

    plt.figure()
    plt.plot(thres, ious_brainwise, color='DarkBlue')
    plt.title('IOU @threshold with {}'.format(method))
    plt.xlabel('Threshold')
    plt.ylabel('IOU')
    plt.grid()

    print('Calculating precision and recall...')
    p = np.array([np.array(mAP(gt_matrix, pred_matrix, frame_num, thr))
                  for thr in thres])
    precision = p[:, 0]
    recall = p[:, 1]
    plt.figure()
    plt.plot(recall, precision, color='DarkBlue')
    plt.title('P-R with {}'.format(method))
    plt.xlabel('Recall')
    plt.ylabel('Precison')
    plt.grid()

    print('Calculating image wise precision and recall...')
    p = np.array([[np.array(ap(gt_matrix[:, :, i], pred_matrix[:, :, i], thr))
                   for i in range(frame_num)] for thr in thres])
    precs = p[:, :, 0]
    recalls = p[:, :, 1]

    precs_df = pd.DataFrame(precs.T, columns=thres)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                 medians='DarkBlue', caps='Gray')
    precs_df.plot.box(grid=True, color=color,
                      title='Precision @threshold with {}'.format(method))
    plt.xlabel('Threshold')
    plt.ylabel('Precision')

    recalls_df = pd.DataFrame(recalls.T, columns=thres)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                 medians='DarkBlue', caps='Gray')
    recalls_df.plot.box(
        grid=True, color=color, title='Recall @threshold with {}'.format(method))
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.show()
