import config
import os

import numpy as np
from model import get_model
from utils import crop_from_img, generate_from_file_list

pos_list = config.pos_list
neg_list = config.neg_list

pos_train = generate_from_file_list(pos_list)
neg_train = generate_from_file_list(neg_list)
print('Pos: {} Neg: {}'.format(len(pos_train), len(neg_train)))

x_train = np.concatenate((pos_train, neg_train), axis=0)
y_train = np.hstack((np.ones(len(pos_train)), np.zeros(len(neg_train))))

# Train
model = get_model(input_shape=(10, 10), output_shape=2)

pos_test = np.load('F:\\GitHub\\DICOM\\coordinate-07\\0710_60day_20190911_135955SWI\\pos.npy')
neg_test = np.load('F:\\GitHub\\DICOM\\testData2\\20190309\\neg.npy')
pos_test = pos_test[:, :, :, np.newaxis]
neg_test = neg_test[:, :, :, np.newaxis]

pos_y = np.ones(len(pos_test))
neg_y = np.zeros(len(neg_test))

model.fit(x_train, y_train, epochs=1000, batch_size=256,
          validation_data=(pos_test, pos_y), validation_freq=100, verbose=0)

## Evaluate
model.evaluate(pos_test, pos_y)
model.evaluate(neg_test, neg_y)
model.evaluate(np.concatenate((pos_test, neg_test), axis=0), np.concatenate((pos_y, neg_y), axis=0))
