import SimpleITK as sitk
import pydicom
import numpy as np
import cv2
import os


def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height


def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    information['NumberOfFrames'] = ds.NumberOfFrames
    return information


def limitedEqualize(img_array, limit=4.0):
    img_array_list = []
    for img in img_array:
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
        img_array_list.append(clahe.apply(img))
    img_array_limited_equalized = np.array(img_array_list)
    return img_array_limited_equalized


def normalize(img_array):
    high = np.max(img_array) * 1.0
    low = np.min(img_array) * 1.0
    img_array = (img_array - low) / (high - low)
    img_array = (img_array * 255).astype('uint8')
    return img_array


# save DICOM files
PathDicom = "SWI/SWI_E5_P1"
DCMFileList = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            # print filename
            DCMFileList.append(os.path.join(dirName, filename))

for filename in DCMFileList:
    img_array, frame_num, width, height = loadFile(filename)
    # print filename
    img_name = filename.split('\\')[-1].split('.')[0] + '.jpg'
    img_array = np.reshape(img_array, (width, height))
    img_array = img_array.astype('uint8')
    img_array_limited_equalized = limitedEqualize(img_array)
    img_array = normalize(img_array)
    cv2.imshow('DICOM', img_array)
    # cv2.imwrite(os.path.join('image/SWI_E5_P1', img_name), img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # print os.path.join('image', img_name)
    cv2.waitKey(0)
