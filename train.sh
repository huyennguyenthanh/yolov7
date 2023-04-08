#!/bin/sh

python train.py --workers 8 --device 0 \
--batch-size 8 \
--data data/havard_polyp.yaml \
--img 640 640 \
--cfg cfg/training/yolov7.yaml \
--weights '' \
--project YOLOv7 \
--name YOLOv7/yolov7_test \
--hyp data/hyp.scratch.p5.yaml \
--epochs 100
