import os

data_root = 'F:\\GitHub\\DICOM\\data\\coordinate-07'

pos_list = ['F:\\GitHub\\DICOM\\data\\testData2\\20190309\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190223\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190222\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190227\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_094509SWI\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_103001SWI\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_110457SWI\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_115808SWI\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_123842SWI\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_131949SWI\\pos.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_135955SWI\\pos.npy', ]

neg_list = ['F:\\GitHub\\DICOM\\data\\testData2\\20190223\\neg.npy',
            'F:\\GitHub\\DICOM\\data\\testData2\\20190222\\neg.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_110457SWI\\neg.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_103001SWI\\neg.npy',
            'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_094509SWI\\neg.npy']

pos_test = 'F:\\GitHub\\DICOM\\data\\coordinate-07\\0710_60day_20190911_143703SWI\\pos.npy'
neg_test = 'F:\\GitHub\\DICOM\\data\\testData2\\20190309\\neg.npy'


def get_path(root=data_root, path='model'):
    model_path = os.path.join(root, path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    return model_path


model_path = get_path(path='model')
