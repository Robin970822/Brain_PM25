import os

data_root = 'F:\\GitHub\\DICOM\\data\\coordinate-07'

pos_list = ['F:\\GitHub\\DICOM\\data\\testData2\\20190309\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190223\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190222\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190227\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_094509SWI\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_103001SWI\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_110457SWI\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_115808SWI\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_123842SWI\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_131949SWI\\pos_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_135955SWI\\pos_p10.npy', ]

neg_list = ['F:\\GitHub\\DICOM\\data\\testData2\\20190223\\neg_p10.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190222\\neg_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_110457SWI\\neg_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_103001SWI\\neg_p10.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_094509SWI\\neg_p10.npy']

pos_test = 'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_143703SWI\\pos_p10.npy'
neg_test = 'F:\\GitHub\\DICOM\\data\\testData2\\20190309\\neg_p10.npy'

pad = 10
data_balance = 2    # 0 for no balance, k for pos:neg = 1:k


def get_path(root=data_root, path='model'):
    model_path = os.path.join(root, path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return model_path


model_path = get_path(path='model')
image_path = get_path(path='image')
result_path = get_path(path='result')
