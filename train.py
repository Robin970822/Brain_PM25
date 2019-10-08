import config
import os
import time
import argparse

import numpy as np
from model import get_model, load_model
from utils import crop_from_img, generate_from_file_list, generate_from_file_list

# args
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input path for model')
parser.add_argument('-o', '--output', help='output path for model',
                    default='{}.h5'.format(time.time()))
parser.add_argument('-m', '--model_type', help='model type', default='FC')
args = parser.parse_args()

# Training Data
pos_list = config.pos_list
neg_list = config.neg_list

pos_train = generate_from_file_list(pos_list)
neg_train = generate_from_file_list(neg_list)

np.random.shuffle(neg_train)
neg_train = neg_train[np.random.choice(
    len(neg_train), int(len(pos_train) * 4))]
print('Pos: {} Neg: {}'.format(len(pos_train), len(neg_train)))

x_train = np.concatenate((pos_train, neg_train), axis=0)
y_train = np.hstack((np.ones(len(pos_train)), np.zeros(len(neg_train))))

# Test Data
pos_test = np.load(
    'F:\\GitHub\\DICOM\\coordinate-07\\0710_60day_20190911_143703SWI\\pos.npy')
neg_test = np.load(
    'F:\\GitHub\\DICOM\\testData2\\20190309\\neg.npy')
pos_test = pos_test[:, :, :, np.newaxis]
neg_test = neg_test[:, :, :, np.newaxis]

pos_y = np.ones(len(pos_test))
neg_y = np.zeros(len(neg_test))

# Train
if args.input:
    model = load_model(args.input)
else:
    model = get_model(input_shape=(10, 10), output_shape=2,
                      model_type=args.model_type)

model.fit(x_train, y_train, epochs=1500, batch_size=256,
          validation_data=(pos_test, pos_y), validation_freq=100, verbose=1)

# Evaluate
model.evaluate(pos_test, pos_y)
model.evaluate(neg_test, neg_y)
model.evaluate(np.concatenate((pos_test, neg_test), axis=0),
               np.concatenate((pos_y, neg_y), axis=0))

model_path = os.path.join(config.model_path, args.output)
model.save(model_path)
print('Model has Saved in {} \n Dataset Pos: {} Neg:{}'.format(
    model_path, len(pos_train), len(neg_train)))
