#!/bin/sh

python test.py --batch-size 64 --task test \
--data data/havard_polyp.yaml \
--weights '/mnt/disk1/nguyen.thanh.huyenb/yolov7_train/YOLOv7/fold_1_percent_5_semi/best.pt' \
--project YOLOv7_test \
--name fold_1_percent_5_semi --benchmark