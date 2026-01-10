#!/bin/bash

#######      chmod +x script_device1.sh
#######      ./script_device1.sh
#######      device1 is 2080ti22g

###260110 小目标中yolo架构消融
yolo detect train data=VisDrone.yaml model=yolo11n-noP5_v1.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-noP5_v1-VD200-d1'
# YOLO11n-noP5_v1 summary: 146 layers, 1,851,556 parameters, 1,851,540 gradients, 5.9 GFLOPs
yolo detect train data=VisDrone.yaml model=yolo11n-noP5_v2.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-noP5_v2-VD200-d1'
# YOLO11n-noP5_v2 summary: 126 layers, 642,708 parameters, 642,692 gradients, 8.3 GFLOPs
yolo detect train data=VisDrone.yaml model=yolo11n-noP5_v3.yaml epochs=200 batch=16 imgsz=640 device=1 name='yolo11n-noP5_v3-VD200-d1'
# YOLO11n-noP5_v3 summary: 161 layers, 897,278 parameters, 897,262 gradients, 9.1 GFLOPs
