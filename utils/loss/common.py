# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale = 2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale/2.0) / self.bin_count
        end = max - (self.scale/2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        #print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float() 
        self.register_buffer('bins', bins) 
               

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result


    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale/2.0) * self.step
        pred_bin = pred[..., 1:(1+self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
    
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0] 
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins) # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE        
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta_RS=0.50, eps=1e-10): 

        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta_RS
        relevant_bg_labels=((targets==0) & (logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error=torch.zeros(fg_num).cuda()
        ranking_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations=fg_logits-fg_logits[ii] 
            bg_relations=relevant_bg_logits-fg_logits[ii]

            if delta_RS > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta_RS)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta_RS)+0.5,min=0,max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)

            # Rank of ii among all examples
            rank=rank_pos+FP_num
                            
            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii]=FP_num/rank      

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations*(1-fg_targets))/rank_pos

            #Find examples in the target sorted order for example ii         
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            #The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            #Compute target sorting error. (Eq. 8)
            #Since target ranking error is 0, this is also total target error 
            target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets))/rank_pos_target

            #Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error
  
            #Identity Update for Ranking Error 
            if FP_num > eps:
                #For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                #For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))

            #Find the positives that are misranked (the cause of the error)
            #These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            #Denominotor of sorting pmf 
            sorting_pmf_denom = torch.sum(missorted_examples)

            #Identity Update for Sorting Error 
            if sorting_pmf_denom > eps:
                #For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                #For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples*(sorting_error[ii]/sorting_pmf_denom))

        #Normalize gradients by number of positives 
        classification_grads[fg_labels]= (fg_grad/fg_num)
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None
    
 
       
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example 
            current_prec=rank_pos/rank[ii]
            
            #Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec<=current_prec):
                max_prec=current_prec
                relevant_bg_grad += (bg_relations/rank[ii])
            else:
                relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
            
            #Store fg gradients
            fg_grad[ii]=-(1-max_prec)
            prec[ii]=max_prec 

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= fg_num
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None
