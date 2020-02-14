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
parser.add_argument('-b', '--method', help='detect method', default='bet')
args = parser.parse_args()

model_name = args.model
clf_path = os.path.join(model_path, model_name)
model = load_model(clf_path)

method = args.method
print('Using {} for detection...'.format(method))
if method == 'bet':
    method_path = os.path.join(model_path, 'unet_BET2.hdf5')
    from detect import detect_file
elif method == 'heatmap':
    method_path = os.path.join(model_path, 'unet_pm25_yuzq.hdf5')
    from detect_by_heatmap import detect_file
elif method == 'unet':
    method_path = os.path.join(model_path, 'unet_pm25_yuzq.hdf5')
    from detect_by_unet import detect_file
elif method == 'aunet':
    method_path = os.path.join(model_path, 'unet_att_500.hdf5')
method_net = unet(pretrained_weights=method_path)

input_path = os.path.join(data_root, args.input)

NII_fileList = []
for dirName, subdirList, fileList in os.walk(input_path):
    for filename in fileList:
        if ".nii" in filename.lower():
            # print filename
            NII_fileList.append(os.path.join(dirName, filename))

for file in tqdm(NII_fileList, desc='Detect in {}'.format(os.path.abspath(input_path))):
    filename = os.path.basename(file)
    output_name = '{}_detect_by_{}.nii'.format(filename.split('.')[-2], method)
    output_path = os.path.join(result_path, output_name)
    mask, _ = detect_file(file, model, method_net)
    nib.save(mask, output_path)
