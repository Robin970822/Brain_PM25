# 文件结构
- config.py:		训练参数设置
- utils.py:			图像处理工具
- data.py:			从nii数据文件中获取训练集和测试集
- model.py:			模型定义及读写
- train.py:			模型训练
- eval.py:			分类模型检验
- detect.py:		检测PM2.5颗粒
- eval_detect.py:	检测模型检验		# TODO

# 实验结果
## 分类结果
|模型|数据集|acc|epoch|
|:---:|:---:|:---:|:---:|
|FC_h3_e1000.h5|Pos: 1518 Neg:1170|Pos: 0.9635 Neg:1.0000|1000|
|CNN_h3_e1000.h5|Pos: 1518 Neg:1170|Pos: 0.9540 Neg:0.9966|1000|
|FC_p10_e1500.h5|Pos: 1500 Neg:6000|Pos: 0.8321 Neg:0.9276|1500|
|CNN_p10_e1500.h5|Pos: 1500 Neg:6000|Pos: 0.8829 Neg:0.9423|1500|
|CNN_p10_e1500_balanced.h5|Pos: 1500 Neg:1500|Pos: 0.9311 Neg:0.7959|1500|
|CNN_p10_e1500_balanced.h5|Pos: 1500 Neg:1500|Pos: 0.9004 Neg:0.8978|1500|

## 检测结果


# DICOM格式文件

DICOM文件，后缀.dcm

读取数据及文件信息样例代码

```python
import SimpleITK as sitk
import pydicom


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
```

# NII格式文件

NII文件，后缀.nii

读取数据样例代码

```python
import nibabel as nib


def loadFile(filename):
    data = nib.load(filename)
    matrix = data.get_data()
    return matrix
```
