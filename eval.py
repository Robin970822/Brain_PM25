import config
import os
import argparse

import numpy as np
from model import load_model

# args
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='model path', default='FC_h3_e1000.h5')
args = parser.parse_args()

model_path = os.path.join(config.model_path, args.path)
model = load_model(model_path)
# Test Data
pos_test = np.load(
    'F:\\GitHub\\DICOM\\coordinate-07\\0710_60day_20190911_143703SWI\\pos.npy')
neg_test = np.load(
    'F:\\GitHub\\DICOM\\testData2\\20190309\\neg.npy')
pos_test = pos_test[:, :, :, np.newaxis]
neg_test = neg_test[:, :, :, np.newaxis]

pos_y = np.ones(len(pos_test))
neg_y = np.zeros(len(neg_test))

# Evaluate
model.evaluate(pos_test, pos_y)
model.evaluate(neg_test, neg_y)
model.evaluate(np.concatenate((pos_test, neg_test), axis=0),
               np.concatenate((pos_y, neg_y), axis=0))