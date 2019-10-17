import config
import os
import argparse

import numpy as np
from model import load_model

# args
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='model path')
args = parser.parse_args()

model_path = os.path.join(config.model_path, args.model_path)
model = load_model(model_path)
# Test Data
pos_test = np.load(config.pos_test)
neg_test = np.load(config.neg_test)
pos_test = pos_test[:, :, :, np.newaxis]
neg_test = neg_test[:, :, :, np.newaxis]

pos_y = np.ones(len(pos_test))
neg_y = np.zeros(len(neg_test))

# Evaluate
model.evaluate(pos_test, pos_y)
model.evaluate(neg_test, neg_y)
model.evaluate(np.concatenate((pos_test, neg_test), axis=0),
               np.concatenate((pos_y, neg_y), axis=0))
