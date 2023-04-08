import torch
import torch.nn as nn
import random
import cv2
import numpy as np
from glob import glob
import os
import sys
sys.path.insert(0, "/home/nguyen.thanh.huyenb/yolov7/yolov7_train")

# Load device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# Load model
from models.common import Conv
def load_model(weight_path):
    model = torch.load(weight_path)['model']

    for m in model.modules():

        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility


    model.eval()
    model.to(device)
    return model

# Load image
from utils.datasets import letterbox
from utils.general import scale_coords, non_max_suppression
def process_image(image_path,stride=32):
    image0 = cv2.imread(image_path)
    image0.shape

    image = letterbox(image0, 320, stride=stride)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device).half()
    image /= 255.0  # 0 - 255 to 0.0 - 1.0
    image = image.unsqueeze(0)

    return image, image0

# Predict
def predict(model, image): 
    with torch.no_grad():
        preds = model(image)[0]
    return preds
def process_output(preds, image, image0):
    names = model.module.names if hasattr(model, 'module') else model.names
    preds = non_max_suppression(preds) # list
  
    pred = preds[0]
    pred[:, :4] = scale_coords(image.shape[2:], pred[:, :4], image0.shape).round()

    for *xyxy, conf, cls in reversed(pred):
        # print(xyxy)
        label = f'{names[int(cls)]} {conf:.2f} '

    return pred

# Visualize
def visualize_results(image_path, image, preds, categories,  visual_folder="/home/nguyen.thanh.huyenb/yolov7/visualize/1"):
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
    visual_path = os.path.join(visual_folder, os.path.basename(image_path))
    bboxes = []
    cls = []
    conf = []

    for p in preds:
        # import pdb; pdb.set_trace()
        bboxes.append(p[:4])
        conf.append(p[4].item())
        cls.append(p[5].item())

       
    for bbox, cf, c in zip(bboxes,conf, cls):
        x0, y0, x1, y1 = bbox[0], bbox[1],bbox[2], bbox[3]
        label = f'{categories[int(c)]} {cf:.2f} '
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 1)
        image = cv2.putText(image, label, (x0 - 2, y0), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 0, 0), 1, cv2.LINE_AA)

    if os.path.exists(label_path):
        h, w = image.shape[:2]
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                h, w = image.shape[:2]
                line = line.split()
                bbox = [float(b) for b in line[1:5]] 
                xc, yc, w, h = [ int(b * s) for b, s in zip(bbox, [w,h,w,h])]
               
               
                label = f'{categories[int(line[0])]}'
                x0 = int(xc - w/2)
                y0 = int(yc - h/2)
                x1 = int(xc + w/2)
                y1 = int(yc + h/2)
                image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (102,255,102), 1)
                image = cv2.putText(image, label, (x0 - 2, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (102,255,102), 1, cv2.LINE_AA)


    cv2.imwrite(visual_path, image)

def save_results(image_path, image, preds, categories, save_folder="/home/nguyen.thanh.huyenb/yolov7/training_minicoco_dataset/test/results"):
    save_filename =  image_path.replace("images", "results").replace(".jpg", ".txt").replace(".png", ".txt")
    with open(save_filename, "w") as f:
        for p in preds:
            x0, y0, x1, y1 = [str(int(i)) for i in p[:4]]
            conf = p[4].item()
            c = p[5].item()
            line = f"{categories[int(c)]} {str(conf)} {x0} {y0} {x1} {y1}\n"
            f.write(line)


# Detect folder of images
def detect(files, model, visualize=False, save_txt=False, visual_folder="/home/nguyen.thanh.huyenb/yolov7/visualize/2"):
    files = [files] if isinstance(files, str) else files
    categories = model.module.names if hasattr(model, 'module') else model.names

    for image_path in files:
        
        image, image0 = process_image(image_path)
        preds = predict(model, image)
        preds = process_output(preds, image, image0)

        if visualize:
            visualize_results(image_path, image0, preds, categories, visual_folder=visual_folder)
        if save_txt:
            save_results(image_path, image0, preds, categories)
            


if __name__ == "__main__":
    files = glob("/home/nguyen.thanh.huyenb/yolov7/training_minicoco_dataset/test/images/*")
    # files = "/home/nguyen.thanh.huyenb/yolov7/polyp_dataset/TestDataset/Kvasir/images/cju16ach3m1da0993r1dq3sn2.png"
    print(f"Found {len(files)} images")

    # load model
    weight_path = "/home/nguyen.thanh.huyenb/yolov7/yolov7_train/runs/train/yolov7-qat-/weights/best.pt"
    model = load_model(weight_path)
    print("Load model successfully")


    names = model.module.names if hasattr(model, 'module') else model.names
    print(f"Found {len(names)} categories: ", names)
    
    visual_folder = "/home/nguyen.thanh.huyenb/yolov7/training_minicoco_dataset/test"
    detect(files, model, save_txt=True)




