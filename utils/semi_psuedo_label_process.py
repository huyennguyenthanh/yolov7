import os
import time
import torch
import torchvision
from .general import *


def non_max_suppression_pseudo_decouple_multi_view(
    prediction,
    conf_thres=0.2,
    cls_thres=0.2,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=None,
    max_det=300,
    multi_view_thres=0.2,
    multi_view_iou_thres=0.7,
):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction[0].shape[2] - 5  # number of classes
    # print(prediction.size())
    # print(prediction[..., 5:].max(-1)[0])
    xc = prediction[0][..., 4] > conf_thres

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output_cls = [torch.zeros((0, 6), device=prediction[0].device)] * prediction[
        0
    ].shape[0]
    output_reg = [torch.zeros((0, 6), device=prediction[0].device)] * prediction[
        0
    ].shape[0]
    for xi, views in enumerate(zip(*prediction)):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = views[0]
        x = x[xc[xi]]  # confidence
        view_scores = []
        view_masks = []
        view_cats = []

        for view in views[1:]:
            view_ = view[view[..., 4] > multi_view_thres]
            i_ = torchvision.ops.nms(
                xywh2xyxy(view_[:, :4]), view_[:, 4], iou_thres
            )  # NMS
            view_ = view_[i_]
            i_ = torchvision.ops.nms(xywh2xyxy(x[:, :4]), x[:, 4], iou_thres)  # NMS
            x = x[i_]
            if view_.size()[0] == 0 or x.size()[0] == 0:
                continue
            view_score = box_iou(xywh2xyxy(x[:, :4]), xywh2xyxy(view_[:, :4]))  # N,M
            view_score, view_ind = view_score.max(-1)
            view_item = view_[view_ind]  # [:,4:]
            view_item[(view_score < multi_view_iou_thres)] *= 0.0
            view_scores.append(view_item.unsqueeze(-1))
            view_masks.append(
                ((view_score >= multi_view_iou_thres))
                .unsqueeze(-1)
                .unsqueeze(-1)
                .float()
            )
            # view_cats.append(((view_item[:,5:].max()[1])==(x[:,5:].max()[1])).unsqueeze(-1).unsqueeze(-1).float())
        view_scores.append(x.unsqueeze(-1))
        view_masks.append(torch.ones_like(x[:, 4]).unsqueeze(-1).unsqueeze(-1).float())
        if len(view_scores):
            view_scores = torch.cat(view_scores, -1).sum(-1)
            view_masks = torch.cat(view_masks, -1).sum(-1)
            view_scores = view_scores / view_masks
            x = view_scores

        """
        传入的unlabel_targets example :(id,classid,x,y,w,h)
       [[0, 6, 0.5, 0.5, 0.26, 0.35],
        [0, 6, 0.5, 0.5, 0.26, 0.35],
        [1, 6, 0.5, 0.5, 0.26, 0.35],
        [2, 6, 0.5, 0.5, 0.26, 0.35], ]
       前两行标签属于第一张图片，第三行属于第二张
       ==> l = [[6, 0.5, 0.5, 0.26, 0.35],
        [ 6, 0.5, 0.5, 0.26, 0.35]] 取第一张
       """

        if labels is not None and len(labels):

            l = labels[(labels[:, 0] == xi).nonzero().squeeze(1)][:, 1:]

            if len(l):
                # Cat apriori labels if autolabelling
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # conf = x[:, 5:].max(1, keepdim=True)[0]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, 4:5]), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 4:5]), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output_boxes = x[i]
        output_reg[xi] = output_boxes[output_boxes[:, 4] > conf_thres][:, :6]
        output_cls[xi] = output_boxes[
            (output_boxes[:, 4] > conf_thres) & (output_boxes[:, 5] > cls_thres)
        ][:, :6]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output_reg, output_cls


def non_max_suppression_pseudo_decouple(
    prediction,
    conf_thres=0.25,
    cls_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=None,
    max_det=300,
):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes

    # xc = prediction[..., 5:].max(-1)[0] > cls_thres  # candidates
    xc = prediction[..., 4] > cls_thres  # candidates
    xf = prediction[..., 4] > conf_thres
    xt = xc | xf
    xs = xc & xf

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output_cls = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_reg = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height

        x = x[xt[xi]]  # confidence
        mask = torch.cat([xc[xi, xt[xi]].unsqueeze(1), xf[xi, xt[xi]].unsqueeze(1)], 1)

        """
        传入的unlabel_targets example :(id,classid,x,y,w,h)
       [[0, 6, 0.5, 0.5, 0.26, 0.35],
        [0, 6, 0.5, 0.5, 0.26, 0.35],
        [1, 6, 0.5, 0.5, 0.26, 0.35],
        [2, 6, 0.5, 0.5, 0.26, 0.35], ]
       前两行标签属于第一张图片，第三行属于第二张
       ==> l = [[6, 0.5, 0.5, 0.26, 0.35],
        [ 6, 0.5, 0.5, 0.26, 0.35]] 取第一张
       """

        if labels is not None and len(labels):

            l = labels[(labels[:, 0] == xi).nonzero().squeeze(1)][:, 1:]
            mask = torch.cat(
                [mask, torch.ones(len(l), 2, dtype=mask.dtype, device=mask.device)], 0
            )
            if len(l):
                # Cat apriori labels if autolabelling
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, j + 5, None], j[:, None].float(), mask[i].float()), 1
            )
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask.float()), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output_x = x[i]
        output_reg[xi] = torch.masked_select(
            output_x[:, :6], output_x[..., -1].unsqueeze(1).bool()
        ).view(-1, 6)
        outbox = torch.masked_select(
            output_x[:, :6], output_x[..., -2].unsqueeze(1).bool()
        ).view(-1, 6)
        output_cls[xi] = outbox
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output_reg, output_cls


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return
