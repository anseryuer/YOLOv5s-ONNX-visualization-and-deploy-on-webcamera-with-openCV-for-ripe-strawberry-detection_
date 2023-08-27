# ripe_strawberry_detection_YOLOv5s

## about the onnx model

detecting ripe strawberry using yolov5

The dataset is not available.

The ipynb file is set on the Google Colab. (Google--my hero)

The model is trained based on the YOLOv5s model and used to detect the ripe strawberries.

The 3 kinds of label, rs10, rs9 and rs8, shows the ripeness of the strawberry.

the final result is shown in the result.jpg.

![result](https://user-images.githubusercontent.com/81740803/183282310-c76a7139-4f24-4b0a-9a08-e35393ab5c2d.jpg)

## visualize.py

After running this file you will see a window for opencv pops out and if a camera on your computer is available.

The model will run in real time(10 frames per second) and detect strawberries and mark their ripeness.

A box will be shown for each object.
