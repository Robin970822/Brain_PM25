import pydicom
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# save DICOM files
PathDicom = "SWI/SWI_E5_P1"
DCMFileList = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            # print filename
            DCMFileList.append(os.path.join(dirName, filename))

# read first file
RefDs = pydicom.read_file(DCMFileList[0])
# build 3D array
PixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(DCMFileList))
PixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
# 3D Data
x = np.arange(0.0, (PixelDims[0]+1)*PixelSpacing[0], PixelSpacing[0])
y = np.arange(0.0, (PixelDims[1]+1)*PixelSpacing[1], PixelSpacing[1])
z = np.arange(0.0, (PixelDims[2]+1)*PixelSpacing[2], PixelSpacing[2])

ArrayDicom = np.zeros(PixelDims, dtype=RefDs.pixel_array.dtype)
for filenameDCM in DCMFileList:
    ds = pydicom.read_file(filenameDCM)
    ArrayDicom[:, :, DCMFileList.index(filenameDCM)] = ds.pixel_array
    plt.figure(dpi=300)
    plt.title(filenameDCM)
    plt.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, DCMFileList.index(filenameDCM)]))
    plt.show()
    cv2.imshow('DICOM', ds.pixel_array)
    cv2.waitKey(0)
