import os
import time
import torch
import copy
import torchvision
import traceback
from .general import *
import copy

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
    certain_conf_thres = 0.4,
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

    

    # certain_cls = []
    # for tensor in output_reg:
    #     if tensor.numel() == 0:
    #         certain_cls.append(torch.zeros((0, 6), device=prediction[0].device))
    #         continue
    #     max_conf_index = torch.argmax(tensor[:, 4])
    #     tensor_with_max_conf = tensor[max_conf_index].unsqueeze(0)
    #     certain_cls.append(tensor_with_max_conf)

    certain_cls = []

    for tensor in output_reg:
        if tensor.numel() == 0:
            certain_cls.append(torch.zeros((0, 6), device=prediction[0].device))
            continue

        conf_above_thres_mask =  tensor[:, 4] > certain_conf_thres

        if torch.any(conf_above_thres_mask):
            certain_cls.append(tensor[conf_above_thres_mask])
        else:
            max_conf_index = torch.argmax(tensor[:, 4])
            tensor_with_max_conf = tensor[max_conf_index].unsqueeze(0)
            certain_cls.append(tensor_with_max_conf)

    return output_reg, certain_cls # output_cls #, certain_reg, certain_cls


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
    Arguments:
        prediction: a tensor of shape (batch_size, num_boxes, num_classes+5), 
        conf_thres: a float value between 0 and 1, representing the confidence threshold for filtering out low-confidence detections based on the objectness score.
        cls_thres: a float value between 0 and 1, representing the confidence threshold for filtering out low-confidence detections based on the class scores.  
        iou_thres: a float value between 0 and 1, representing the IoU (Intersection over Union) threshold for suppressing overlapping detections.
        classes: a list of integers representing the indices of the object classes to consider during NMS. If not specified, all classes will be considered.
        agnostic: a boolean value indicating whether to perform class-agnostic NMS. If True, objectness and class scores will be considered separately during NMS.
        multi_label: a boolean value indicating whether to allow multiple labels per box. If True, the class with the highest score will be used as the label.
        labels: a tensor of shape (num_boxes, 6), where each row corresponds to a ground truth bounding box in the format (image_index, class_index, x_center, y_center, width, height). If specified, these labels will be used to perform auto-labeling during NMS.
        max_det: an integer value indicating the maximum number of detections to output per image.
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
    min_wh, max_wh = 10, 4096  # (pixels) minimum and maximum box width and height
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

    

    return output_reg,  output_cls #, certain_reg, certain_cls



def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def convert_output_to_label(imgs, preds, shapes, conf=False, device=torch.device('cuda:0')):
    """
        x,y,x,y, conf, label -> id, label, x,y,w,h,
    """
    if conf:
        labels = torch.zeros(0, 7).to(device)
    else:
        labels = torch.zeros(0, 6).to(device)
    
    for i, _pred in enumerate(preds):
        pred = copy.deepcopy(_pred)
        if conf:
            label = torch.zeros(len(pred), 7).to(device)
        else:
            label = torch.zeros(len(pred), 6).to(device)
        h0, w0  = shapes[i][0]
        box = scale_coords(imgs[i].shape[:2], pred[:, 0:4], shapes[i][0], shapes[i][1]).round()
        box = xyxy2xywhn(pred[:, 0:4], w0, h0)
        label[:, 0] = i
        label[:, 1] = pred[:, 5]
        label[:, 2:6] = box
        if conf:
            label[:, 6] = pred[:, 4]
        labels = torch.cat([labels, label])

    return labels


def convert_output_to_label_2(imgs, preds, shapes=None, conf=False, device=torch.device('cuda:0')):
    if conf:
        unlabel_targets_merge = torch.zeros(0, 7).to(device)
    else:
        unlabel_targets_merge = torch.zeros(0, 6).to(device)

    semi_label_items = torch.zeros(1, device=device)
    for batch_ind, pseudo_box in enumerate(preds):
        # two stage filters
        n_box = pseudo_box.size()[0]
        if conf:
            unlabel_target = torch.zeros(n_box, 7).to(device)
        else:
            unlabel_target = torch.zeros(n_box, 6).to(device)
        unlabel_target[:, 0] = batch_ind
        unlabel_target[:, 1] = pseudo_box[:, -1]
        unlabel_target[:, 2:6] = xyxy2xywhn(pseudo_box[:, 0:4], w=imgs.size()[2],
                                            h=imgs.size()[3])
        if conf:
            unlabel_target[:, 6] = pseudo_box[:, 4]
        unlabel_targets_merge = torch.cat([unlabel_targets_merge, unlabel_target])

        semi_label_items += n_box
    
    return unlabel_targets_merge
                       
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # build candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # get candidates based on confidence score

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

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
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def non_max_suppression_custom(predictions, conf_thres=0.25, iou_thres=0.45):
    """Custom Non-Maximum Suppression (NMS) implementation

    Returns:
         Two lists of detections: output_cls (high confidence scores), output_reg (high box and iou scores)
    """
    output_cls = []  # list for detections with high confidence scores
    output_reg = []  # list for detections with high box and iou scores

    for prediction in predictions:
        # Filter detections based on confidence threshold
        mask = prediction[..., 4] > conf_thres
        filtered_pred = prediction[mask]

        if filtered_pred.shape[0] == 0:
            continue

        # Compute box scores
        box_scores = filtered_pred[..., 4]

        # Apply NMS
        keep = torchvision.ops.nms(filtered_pred[..., :4], box_scores, iou_threshold=iou_thres)

        # Split detections into cls and reg based on scores
        output_cls.append(filtered_pred[keep])
        output_reg.append(filtered_pred[keep])

    return output_cls, output_reg


def convert_to_eval_output(model,out,device=torch.device("cpu")):
    try:
        z = []
        no = model.nc + 5
        bs = out[0].shape[0]
        for i in range(len(out)):

            y = out[i].sigmoid().to(device)
            
            if model.model[-1].grid[i].shape[2:4] != out[i].shape[2:4]:
                ny, nx = y.shape[2:4]
                model.model[-1].grid[i] = model.model[-1]._make_grid(nx, ny).to(out[i].device)

            
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + model.model[-1].grid[i].to(device)) * model.model[-1].stride[i].to(device)  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * model.model[-1].anchor_grid[i]  # wh
            z.append(y.view(bs, -1, no))
        z = torch.cat(z, 1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        import pdb; pdb.set_trace()
    return z