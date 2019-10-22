import config
import os
import time
import argparse

import numpy as np

from sklearn.utils import class_weight
from model import get_model, load_model
from utils import generate_from_file_list

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

np.random.shuffle(neg_train)

if not config.data_balance == 0:
    neg_train = neg_train[np.random.choice(len(neg_train), int(
        len(pos_train) * config.data_balance * np.random.uniform(0.5, 1.5)))]

print('\nPos: {} Neg: {}\n'.format(len(pos_train), len(neg_train)))

x_train = np.concatenate((pos_train, neg_train), axis=0)
y_train = np.hstack((np.ones(len(pos_train)), np.zeros(len(neg_train))))

# Test Data
pos_test = np.load(config.pos_test)
neg_test = np.load(config.neg_test)
pos_test = pos_test[:, :, :, np.newaxis]
neg_test = neg_test[:, :, :, np.newaxis]

pos_y = np.ones(len(pos_test))
neg_y = np.zeros(len(neg_test))

print('\nReading Data Done.\n')

# Train
# Load Model
print('\nTrainging Begin\n')
print('Loading Model...')
if args.input:
    model = load_model(args.input)
else:
    model = get_model(input_shape=(20, 20), output_shape=2,
                      model_type=args.model)
# Train
print('Trainging...')

class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(y_train), y_train)
print(class_weights)

model.fit(x_train, y_train, epochs=args.epochs, batch_size=256, class_weight=class_weights,
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
