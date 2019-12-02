from detect import detect_file
from model import load_model, unet
from tqdm import tqdm

import os
import config
import argparse

import nibabel as nib

data_root = config.data_root
model_path = config.model_path
result_path = config.result_path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input path', default='../data0927')
parser.add_argument('-m', '--model', help='model path',
                    default='CNN_p10_e2000.h5')
args = parser.parse_args()

model_name = args.model
bet_path = os.path.join(model_path, 'unet_BET2.hdf5')
clf_path = os.path.join(model_path, model_name)
bet_net = unet(pretrained_weights=bet_path)
model = load_model(clf_path)

input_path = os.path.join(data_root, args.input)

NII_fileList = []
for dirName, subdirList, fileList in os.walk(input_path):
    for filename in fileList:
        if ".nii" in filename.lower():
            # print filename
            NII_fileList.append(os.path.join(dirName, filename))

for file in tqdm(NII_fileList, desc='Detect in {}'.format(os.path.abspath(input_path))):
    filename = os.path.basename(file)
    output_name = '{}_detect_bet.nii'.format(filename.split('.')[-2])
    output_path = os.path.join(result_path, output_name)
    mask, _ = detect_file(file, model, bet_net)
    nib.save(mask, output_path)
