
import sys
sys.path.insert(0, "../")  
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU

# Load PyTorch model
weights = "/home/nguyen.thanh.huyenb/yolov7/yolov7/runs/train/yolov79/weights/best.pt"
weights = "/home/nguyen.thanh.huyenb/yolov7/yolov7_train/runs/train/yolov7-havard-polyp-1-label-/weights/best.pt"
save_filename = "/home/nguyen.thanh.huyenb/yolov7/yolov7/yolov79_best.onnx"
save_filename = "/home/nguyen.thanh.huyenb/yolov7/serve-yolov7/weights/yolov7-polyp/1/best.onnx"
# device = torch.device('cuda:0')
model = attempt_load(weights, map_location="cpu")  # load FP32 model

# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
        if isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
        elif isinstance(m.act, nn.SiLU):
            m.act = SiLU()
model.model[-1].export = False  # 为fasle的话，就将输出的三个xcat。否则不会将输出cat
model.eval()

# Input
img = torch.randn(1, 3, 224, 320)  # image size(1,3,320,192) iDetection
dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},  # size(1,3,640,640)
                "output": {0: "batch", 2: "y", 3: "x"},
            }
torch.onnx.export(
    model,
    img,
    save_filename,
    verbose=False,
    opset_version=12,
    input_names=["images"],
    output_names=["outputs"],
    dynamic_axes=dynamic_axes

)
print("Successfully")
