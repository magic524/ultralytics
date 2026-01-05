#!/bin/bash

#######      chmod +x script_device1.sh
#######      ./script_device1.sh
#######      device1 is 2080ti22g

# yolo detect train data=VisDrone.yaml model=yolo11n-CAAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-CAAM-VD200-d1'
# yolo detect train data=VisDrone.yaml model=yolo11n-SSAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-SSAM-VD200-d1'

yolo detect train data=VisDrone.yaml model=yolo11n.yaml epochs=200 batch=16 imgsz=640 device=1 name='v11n_CIoU-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11s.yaml epochs=200 batch=16 imgsz=640 device=1 name='v11s_CIoU-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11m.yaml epochs=200 batch=8 imgsz=640 device=1 name='v11m_CIoU-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11l.yaml epochs=200 batch=4 imgsz=640 device=1 name='v11l_CIoU-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11x.yaml epochs=200 batch=4 imgsz=640 device=1 name='v11x_CIoU-VD200-d1'