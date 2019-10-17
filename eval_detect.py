import config
import os
import argparse

import numpy as np
import nibabel as nib

data_root = config.data_root
result_path = config.result_path

# args
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict', help='predict for detection')
parser.add_argument('-g', '--groundtruth',
                    help='ground truth for detection', default='07-143m.nii')
args = parser.parse_args()

# predict and ground truth path
predict_path = args.predict
predict_path = os.path.join(result_path, predict_path)

groundtruth_path = args.groundtruth
groundtruth_path = os.path.join(data_root, groundtruth_path)

# read data from path
predict = nib.load(predict_path)
groundtruth = nib.load(groundtruth_path)
predict = predict.get_data()
groundtruth = groundtruth.get_data()

predict[predict > 0] = 1

tp = np.sum(np.array(predict, dtype=np.uint8) & groundtruth)
print(tp)
