import SimpleITK as sitk
import numpy as np
import cv2 as cv
import os
import desull as de


def convert_from_dicom_to_jpg(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window*1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])    #归一化
    newimg = (newimg*255).astype('uint8')                #将像素值扩展到[0,255]
    cv.imwrite(save_path, newimg, [int(cv.IMWRITE_JPEG_QUALITY), 100])



def file_name(file_dir):
    L=[]
    print("打开成功")
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file[-4:] == '.dcm':
                L.append(os.path.join(root, file))
    # print(L[1:])
    # print(len(L))
    return L
# file_name("E:\softwareGD\dataset\TCGA-GBM")



def batches(dictory):
    output=[]
    for i in dictory:
        dic=""
        L=[]
        L=i.split("\\")
        #L[4]="Brain_tumor_jpg"
        #print(L[-1][-4:])
        L[-1] =L[-1][:-4]+ ".jpg"
        for j in range(0,len(L)):
            dic += L[j]+"//"

        output.append(dic[:-2])
    return output



def process(dictory):
    for i in range(0,len(dictory)):
        # 下面是将对应的dicom格式的图片转成jpg
        print(dictory[i])
        output=batches(dictory)
        dcm_image_path = dictory[i]      #读取dicom文件
        output_jpg_path = output[i]
        ds_array = sitk.ReadImage(dcm_image_path)         #读取dicom文件的相关信息
        img_array = sitk.GetArrayFromImage(ds_array)      #获取array
        # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
        #类似于 （1，height，width）的形式
        shape = img_array.shape
        #print(shape)
        #print(img_array)
        img_array = np.reshape(img_array, (shape[1], shape[2]))  #获取array中的height和width
        #print(img_array)
        high = np.max(img_array)
        low = np.min(img_array)
        convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)   #调用函数，转换成jpg文件并保存到对应的路径
        print('FINISHED'+dictory[i]+"\n")
        print(output[i])

process(file_name("E:\softwareGD\dataset\TCGA-GBM"))

def processDeskull(dictory):
    for i in range(0,len(dictory)):
        output=batches(dictory)
        output_jpg_path = output[i]
        src = cv.imread(dictory[i])
        new = de.watershed_demo(src)
        cv.imshow("cnm",new)
        cv.imwrite(output_jpg_path,new)
        print('FINISHED'+dictory[i]+"\n")
        print(output[i])

# processDeskull(file_name("E:\softwareGD\dataset\TCGA-GBM"))



