# DICOM格式文件

DICOM文件，后缀.dcm

## 读取

### 导入主要框架：SimpleTK、pydicom、PIL、cv2和numpy

```python
import SimpleITK as sitk
from PIL import Image
import pydicom
import numpy as np
import cv2
```

# 实验结果
|模型|数据集|acc|epoch|
|:---:|:---:|:---:|:---:|
|FC_h3_e1000.h5|Pos: 1518 Neg:1170|Pos: 0.9635 Neg:1.0000|1000|
|CNN_h3_e1000.h5|Pos: 1518 Neg:1170|Pos: 0.9540 Neg:0.9966|1000|


