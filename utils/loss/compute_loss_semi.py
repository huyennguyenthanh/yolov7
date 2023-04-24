import torch
import torch.nn as nn
import torch.nn.functional as F

from ..general import (
    bbox_iou,
    bbox_alpha_iou,
    box_iou,
    box_giou,
    box_diou,
    box_ciou,
    xywh2xyxy,
)
from ..torch_utils import is_parallel
from .common import *


class ComputeLossSemi:
    def __init__(self) -> None:
        pass
