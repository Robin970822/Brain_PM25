import os
import json


with open('config.json', 'r') as jobj:
    config_dict = json.load(jobj)

data_root = config_dict['data_root']

# training data
pos_list = config_dict['pos_list']
neg_list = config_dict['neg_list']

# validate data
pos_test = config_dict['pos_test']
neg_test = config_dict['neg_test']

pad = config_dict['pad']
data_balance = config_dict['data_balance']


def get_path(root=data_root, path='model'):
    model_path = os.path.join(root, path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return model_path


model_path = get_path(path='model')
image_path = get_path(path='image')
result_path = get_path(path='result')

bet_path = os.path.join(model_path, 'unet_BET2.hdf5')
