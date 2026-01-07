#!/bin/bash

#######      chmod +x script_device1.sh
#######      ./script_device1.sh
#######      device1 is 2080ti22g

#yolo detect train data=VisDrone.yaml model=yolo11n-CAAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-CAAM-VD200-d1'
#yolo detect train data=VisDrone.yaml model=yolo11n-SSAM.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-SSAM-VD200-d1'

# Train CAAMv3 by default; keep CAAMv1 and CAAMv2 as references
# yolo detect train data=VisDrone.yaml model=yolo11n-CAAMv1.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-CAAMv1-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11n-CAAMv2.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-CAAMv2-VD200-d1'
yolo detect train data=VisDrone.yaml model=yolo11n-CAAMv3.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-CAAMv3-VD200-d1'
#yolo detect train data=VisDrone.yaml model=yolo11n-SSAMv1.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-SSAMv1-VD200-d1'