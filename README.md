# ripe_strawberry_detection_YOLOv5s

Detecting the intended object in ONNX model. When running the python file, a window will pop up on the computer marking the input from the camera with squares.

The current model is for strawberry detection.

## about the onnx model

detecting ripe strawberry using yolov5

The model is trained based on the YOLOv5s model and used to detect the ripe strawberries.

The 3 kinds of label, rs10, rs9 and rs8, shows the ripeness of the strawberry.

the final result is shown in the result.jpg.

![result](https://user-images.githubusercontent.com/81740803/183282310-c76a7139-4f24-4b0a-9a08-e35393ab5c2d.jpg)

## visualize.py

After running this file you will see a window for opencv pops out and if a camera on your computer is available.

The model will run in real time(10 frames per second) and detect strawberries and mark their ripeness.

A box will be shown for each object.

## Result

![confusion_matrix](https://github.com/anseryuer/YOLOv5s-ONNX-visualization-and-deploy-on-webcamera-with-openCV-for-ripe-strawberry-detection_/assets/81740803/2a9b6c17-22bf-4929-8a4f-b1ef40740e36)

![PR_curve](https://github.com/anseryuer/YOLOv5s-ONNX-visualization-and-deploy-on-webcamera-with-openCV-for-ripe-strawberry-detection_/assets/81740803/37c7fe5c-b169-4fb9-aad6-d9f325a607d6)

