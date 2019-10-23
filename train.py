import config
import os
import time
import argparse

import numpy as np

from model import get_model, load_model
from utils import generate_from_file_list, random_shuffle, augmentation

# args
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input path for model')
parser.add_argument('-o', '--output', help='output path for model',
                    default='{}.h5'.format(time.time()))
parser.add_argument('-m', '--model', help='model type', default='CNN')
parser.add_argument(
    '-e', '--epochs', help='number of training epochs', default=1500, type=int)
args = parser.parse_args()

print('\nReading Data...\n')
# Training Data
pos_list = config.pos_list
neg_list = config.neg_list

pos_train = generate_from_file_list(pos_list)
neg_train = generate_from_file_list(neg_list)

np.random.shuffle(pos_train)
np.random.shuffle(neg_train)

if not config.data_balance == 0:
    neg_train = neg_train[np.random.choice(len(neg_train), int(
        len(pos_train) * config.data_balance))]

print('\nPos: {} Neg: {}\n'.format(len(pos_train), len(neg_train)))

x_train = np.concatenate((pos_train, neg_train), axis=0)
y_train = np.hstack((np.ones(len(pos_train)), np.zeros(len(neg_train))))

# shuffle and augmentation
x_train, y_train = random_shuffle(x_train, y_train)
x_train = augmentation(x_train)

# Test Data
pos_test = generate_from_file_list([config.pos_test])
neg_test = generate_from_file_list([config.neg_test])

pos_y = np.ones(len(pos_test))
neg_y = np.zeros(len(neg_test))

print('\nReading Data Done.\n')

# Train
# Load Model
print('\nTrainging Begin\n')
print('Loading Model...')
pad = config.pad
if args.input:
    model = load_model(args.input)
else:
    model = get_model(input_shape=(2*pad, 2*pad), output_shape=2,
                      model_type=args.model)
# Train
print('Trainging...')

model.fit(x_train, y_train, epochs=args.epochs, batch_size=256,
          validation_data=(pos_test, pos_y), validation_freq=100, verbose=1)

# Evaluate
print('Evaluating...')
model.evaluate(pos_test, pos_y)
model.evaluate(neg_test, neg_y)
model.evaluate(np.concatenate((pos_test, neg_test), axis=0),
               np.concatenate((pos_y, neg_y), axis=0))

# Save Model
print('Saving Model...')
model_path = os.path.join(config.model_path, args.output)
model.save(model_path)
print('Model has Saved in {} \n Dataset Pos: {} Neg:{}'.format(
    model_path, len(pos_train), len(neg_train)))
