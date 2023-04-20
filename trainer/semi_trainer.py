from copy import deepcopy
from formatter import test
import os
from threading import Thread
import time
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.cuda import amp

import yaml
from models.yolo import Model
from trainer.trainer import Trainer
from utils.datasets import create_dataloader
import test
from utils.general import (
    check_dataset,
    check_file,
    check_img_size,
    colorstr,
    init_seeds,
    labels_to_class_weights,
    one_cycle,
    increment_path,
    strip_optimizer,
)
from utils.loss.compute_loss_ota import ComputeLossOTA
from utils.loss.compute_loss import ComputeLoss
from utils.loss.compute_loss_semi import ComputeLossSemi
from utils.metrics import fitness
from utils.plots import plot_images, plot_results
from utils.semi_psuedo_label_process import (
    non_max_suppression_pseudo_decouple,
    non_max_suppression_pseudo_decouple_multi_view,
    xyxy2xywhn,
)
from utils.torch_utils import (
    ModelEMA,
    _update_teacher_model,
    torch_distributed_zero_first,
)
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


class SemiTrainer(Trainer):
    def __init__(self, opt, device) -> None:

        """_summary_

        Args:
            opt (dict):

            Namespace(semi=False, weights='',
            cfg='cfg/training/yolov7.yaml',
            data='data/havard_polyp.yaml',
            hyp='data/hyp.scratch.p5.yaml',
            epochs=2, batch_size=16,
            img_size=[640, 640], rect=False,
            resume=False, nosave=False, notest=False,
            noautoanchor=False, evolve=False,
            bucket='', cache_images=False,
            image_weights=False,
            device='0',
            multi_scale=False, single_cls=False,
            adam=False, sync_bn=False,
            local_rank=-1, workers=8,
            project='YOLOv7_test', entity=None, name='yolov7_test_refactor',
            exist_ok=False, quad=False,
            linear_lr=False,
            label_smoothing=0.0,
            upload_dataset=False, bbox_interval=-1, s
            ave_period=-1, artifact_alias='latest',
            freeze=[0], v5_metric=False,
            world_size=1, global_rank=-1)
            device (_type_): _description_
            LOCAL_RANK (_type_): _description_
            RANK (_type_): _description_
            WORLD_SIZE (_type_): _description_
        """
        self.opt = opt
        self.device = device
        self.rank = opt.global_rank

        # Set up env
        self.resume()
        self.check_DDP_mode()
        self.hyp = self.load_hyperparameter()

        # Set up logging and saving
        self.opt.save_dir = increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve
        )
        self.save_dict, self.data_dict = self.create_save_folder()
        self.wandb_logger, self.data_dict = self.log_wandb(self.data_dict)
        self.nc = int(self.data_dict["nc"])  # number of classes
        self.names = self.data_dict["names"]  # class names
        self.nominal_batch_size = 64  # nominal batch size - nbs
        self.accumulate = max(
            round(self.nominal_batch_size / self.opt.total_batch_size), 1
        )  # the number of iterations to accumulate gradients before performing a weight update in (SGD)
        self.hyp["weight_decay"] *= (
            self.opt.total_batch_size * self.accumulate / self.nominal_batch_size
        )  # scale weight_decay
        logger.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")
        assert len(self.names) == self.nc, "%g names found for nc=%g dataset in %s" % (
            len(self.names),
            self.nc,
            opt.data,
        )  # check

        # Load Models
        self.teacher_model = self.load_model()
        self.student_model = self.load_model()
        self.optimizer = self.load_optimizer()
        (
            self.scheduler,
            self.lf,
        ) = self.load_scheduler()  # scheduler, learning rate lambda

        # Load dataset
        self.gs = max(int(self.teacher_model.stride.max()), 32)  # grid zise
        self.num_layers = self.teacher_model.model[
            -1
        ].nl  # number of detection layers (used for scaling hyp['obj'])
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in opt.img_size]
        (
            self.labeled_dataset,
            self.labeled_dataloader,
            self.labeled_num_batches,
        ) = self.load_dataset(type="train")
        (
            self.unlabeled_dataset,
            self.unlabeled_dataloader,
            self.unlabeled_num_batches,
        ) = self.load_dataset(type="unlabel")
        (
            self.val_dataset,
            self.val_dataloader,
            self.val_num_batches,
        ) = self.load_val_dataset()
        self.num_batches = max(self.labeled_num_batches, self.unlabeled_num_batches)
        # EMA
        self.ema = ModelEMA(self.student_model) if self.rank in [-1, 0] else None

        # Model parameters
        self.hyp["box"] *= 3.0 / self.num_layers  # scale to layers
        self.hyp["cls"] *= (
            self.nc / 80.0 * 3.0 / self.num_layers
        )  # scale to classes and layers
        self.hyp["obj"] *= (
            (self.imgsz / 640) ** 2 * 3.0 / self.num_layers
        )  # scale to image size and layers
        self.hyp["label_smoothing"] = opt.label_smoothing
        self.student_model.nc = self.nc  # attach number of classes to model
        self.student_model.hyp = self.hyp  # attach hyperparameters to model
        self.student_model.class_weights = (
            labels_to_class_weights(self.labeled_dataset.labels, self.nc).to(device)
            * self.nc
        )  # attach class weights
        self.student_model.names = self.names

        self.time_dict = {}

    def load_dataset(self, type="train"):
        opt = self.opt
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)  # check
        path = self.data_dict["train"]

        # Trainloader
        dataloader, dataset = create_dataloader(
            path,
            self.imgsz,
            opt.batch_size,
            self.gs,
            opt,
            hyp=self.hyp,
            augment=True,
            cache=opt.cache_images,
            rect=opt.rect,
            rank=self.rank,
            world_size=opt.world_size,
            workers=opt.workers,
            image_weights=opt.image_weights,
            quad=opt.quad,
            prefix=colorstr("train: "),
        )
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert (
            mlc < self.nc
        ), "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
            mlc,
            self.nc,
            opt.data,
            self.nc - 1,
        )
        return dataset, dataloader, nb

    def load_val_dataset(self):
        opt = self.opt
        if self.rank in [-1, 0]:
            dataset, dataloader = create_dataloader(
                self.data_dict["val"],
                self.imgsz_test,
                opt.batch_size * 2,
                self.gs,
                opt,  # testloader
                hyp=self.hyp,
                cache=opt.cache_images and not opt.notest,
                rect=True,
                rank=-1,
                world_size=opt.world_size,
                workers=opt.workers,
                pad=0.5,
                prefix=colorstr("val: "),
            )
        nb = len(dataloader)
        return dataset, dataloader, nb

    def train(self):
        print("Hello")

        self.num_warm_up_iters = max(
            round(self.hyp["warmup_epochs"] * self.num_batches), 1000
        )  # number of warmup iterations, max(3 epochs, 1k iterations)
        self.maps = np.zeros(self.nc)  # mAP per class
        self.results_list = [
            (0, 0, 0, 0, 0, 0, 0)
        ]  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        start_epoch, self.best_fitness = 0, 0.0
        self.scheduler.last_epoch = start_epoch - 1
        self.scaler = amp.GradScaler(enabled=self.device)

        self.compute_loss_semi = ComputeLossOTA(self.student_model)  # init loss class
        self.compute_loss = ComputeLoss(self.student_model)  # init loss class

        for epoch in range(start_epoch, self.opt.epochs):
            self.train_on_epoch(epoch)
        self.end_training()

    def train_on_epoch(self, epoch):
        self.student_model.train()
        mean_loss = torch.zeros(4, device=self.device)  # mean losses
        mean_loss_semi = torch.zeros(3, device=self.device)  # mean losses
        semi_loss_items = torch.zeros(3, device=self.device)
        semi_label = torch.zeros(1, device=self.device)

        if self.rank != -1:
            self.labeled_dataloader.sampler.set_epoch(epoch)
            self.unlabeled_dataloader.sampler.set_epoch(epoch)
            self.labeled_dataloader.set_length(
                max(len(self.labeled_dataloader), len(self.unlabeled_dataloader))
            )
            self.unlabeled_dataloader.set_length(
                max(len(self.labeled_dataloader), len(self.inlabeled_dataloader))
            )
        pbar = enumerate(zip(self.labeled_dataloader, self.unlabeled_dataloader))
        logger.info(
            ("\n" + "%10s" * 11)
            % (
                "Epoch",
                "gpu_mem",
                "box",
                "obj",
                "cls",
                "mls",
                "semi_obj",
                "semi_cls",
                "semi_mls",
                "labels",
                "semi_labels",
            )
        )
        pbar = tqdm(pbar, total=self.num_batches)  # progress bar
        self.optimizer.zero_grad()

        for i, data in pbar:

            ni = i + self.num_batches * epoch  # num iteration
            imgs = (
                imgs.to(self.device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0
            # Warm up --------------------------------
            if ni <= self.num_warm_up_iters:
                self.warm_up(epoch, ni)

            self.train_on_batch(ni, data)

    def train_on_batch(self, ni, data):
        semi_label_items = torch.zeros(1, device=self.device)
        (label_imgs, label_targets, label_class_one_hot, _, _), (
            unlabel_imgs,
            unlabel_targets,
            unlabel_class_one_hot,
            _,
            _,
        ) = data

        label_imgs_weak_aug, label_imgs_strong_aug = label_imgs
        unlabel_imgs_weak_aug, unlabel_imgs_strong_aug = unlabel_imgs
        label_imgs_weak_aug = label_imgs_weak_aug.to(
            self.device, non_blocking=True
        ).float()
        label_imgs_strong_aug = label_imgs_strong_aug.to(
            self.device, non_blocking=True
        ).float()
        unlabel_imgs_weak_aug = unlabel_imgs_weak_aug.to(
            self.device, non_blocking=True
        ).float()
        unlabel_imgs_strong_aug = unlabel_imgs_strong_aug.to(
            self.device, non_blocking=True
        ).float()
        label_targets = label_targets.to(self.device, non_blocking=True)
        unlabel_targets = unlabel_targets.to(self.device, non_blocking=True)

        label_imgs = torch.cat([label_imgs_weak_aug, label_imgs_strong_aug], 0)
        label_class_one_hot = torch.cat(
            [label_class_one_hot, label_class_one_hot.detach().clone()], 0
        )
        label_targets_strong = label_targets.clone().detach()

        if label_targets.size()[0] == 0 or label_targets_strong.size()[0] == 0:
            print(label_imgs_weak_aug.size())
            return
        label_targets_strong[:, 0] += label_targets[-1, 0] + 1
        label_targets = torch.cat([label_targets, label_targets_strong], 0)


        semi_loss = 0.0
        if ni < self.hyp["burn_up_step"]:
            loss = self.train_on_batch_supervised(
                ni, label_imgs, label_targets, label_class_one_hot
            )
        else:
            self.update_ema(ni)
            (
                unlabel_class_one_hot,
                unlabel_targets_merge_cls,
                unlabel_targets_merge_reg,
            ) = self.predict_pseudo_label(unlabel_imgs_weak_aug)
            loss, semi_loss, semi_loss_items = self.train_on_batch_semi(
                label_imgs,
                label_targets,
                unlabel_imgs_strong_aug,
                label_class_one_hot,
                unlabel_class_one_hot,
                unlabel_targets_merge_cls,
                unlabel_targets_merge_reg,
            )

        loss = loss + semi_loss * self.hyp["semi_loss_weight"]
        semi_loss_items *= self.hyp["semi_loss_weight"]
        # Backward
        self.scaler.scale(loss).backward()

        # Optimize
        if ni - last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema is not None:
                self.ema.update(self.student_model )
            last_opt_step = ni

        self.plot_on_batch()

    def train_on_batch_supervised(
        self, ni, label_imgs, label_targets, label_class_one_hot
    ):

        
        if ni < self.hyp["burn_up_step"]:
            # Forward
            with amp.autocast(enabled=self.device):
                pred, pred_mls = self.student_model(label_imgs)  # forward
                loss, loss_items = self.compute_loss(
                    pred,
                    label_targets.to(self.device),
                    pred_mls,
                    label_class_one_hot.to(self.device),
                )  # loss scaled by batch_size
                if self.opt.quad:
                    loss *= 4.0
        return loss

    def update_ema(self, ni):
        if ni == self.hyp["burn_up_step"]:
            _update_teacher_model(
                self.student_model if self.ema is None else self.ema.ema,
                self.teacher_model,
                keep_rate=0.0,
            )
        elif (ni - self.hyp["burn_up_step"]) % self.hyp["teacher_update_iter"] == 0:
            self._update_teacher_model(
                self.student_model if self.ema is None else self.ema.ema,
                self.teacher_model,
                keep_rate=self.hyp["ema_keep_rate"],
            )

    def predict_pseudo_label(self, unlabel_imgs_weak_aug):

        # Predict
        self.teacher_model.eval()
        with torch.no_grad():
            out, train_out, pseudo_class_one_hot = self.teacher_model(
                unlabel_imgs_weak_aug, augment=True
            )
            pseudo_class_one_hot_post = (
                pseudo_class_one_hot.detach().sigmoid() > self.hyp["mls_threshold"]
            ).float()

            if isinstance(out, list):
                (
                    pseudo_boxes_reg,
                    pseudo_boxes_cls,
                ) = non_max_suppression_pseudo_decouple_multi_view(
                    out,
                    self.hyp["bbox_threshold"],
                    self.hyp["cls_threshold"],
                    multi_label=True,
                    agnostic=True,
                )
            else:

                (
                    pseudo_boxes_reg,
                    pseudo_boxes_cls,
                ) = non_max_suppression_pseudo_decouple(
                    out,
                    self.hyp["bbox_threshold"],
                    self.hyp["cls_threshold"],
                    multi_label=True,
                    agnostic=True,
                )

        unlabel_class_one_hot = pseudo_class_one_hot_post.clone().detach()
        unlabel_targets_merge_reg = torch.zeros(0, 6).to(self.device)
        unlabel_targets_merge_cls = torch.zeros(0, 6).to(self.device)
        # Filter
        for batch_ind, (pseudo_box_reg, pseudo_box_cls, pseudo_one_hot) in enumerate(
            zip(pseudo_boxes_reg, pseudo_boxes_cls, pseudo_class_one_hot_post)
        ):
            unlabel_target_probs = torch.gather(
                pseudo_one_hot.sigmoid(), 0, pseudo_box_cls[:, -1].long()
            )
            unlabel_target_vaild = (
                unlabel_target_probs > self.hyp["mls_filter_threshold"]
            )
            pseudo_box_cls = torch.masked_select(
                pseudo_box_cls, unlabel_target_vaild.unsqueeze(1)
            ).view(-1, 6)

            n_box = pseudo_box_cls.size()[0]
            unlabel_target_cls = torch.zeros(n_box, 6).to(self.device)
            unlabel_target_cls[:, 0] = batch_ind
            unlabel_target_cls[:, 1] = pseudo_box_cls[:, -1]
            unlabel_target_cls[:, 2:] = xyxy2xywhn(
                pseudo_box_cls[:, 0:4],
                w=unlabel_imgs_weak_aug.size()[2],
                h=unlabel_imgs_weak_aug.size()[3],
            )
            unlabel_targets_merge_cls = torch.cat(
                [unlabel_targets_merge_cls, unlabel_target_cls]
            )

            n_box = pseudo_box_reg.size()[0]
            unlabel_target_reg = torch.zeros(n_box, 6).to(self.device)
            unlabel_target_reg[:, 0] = batch_ind
            unlabel_target_reg[:, 1] = pseudo_box_reg[:, -1]
            unlabel_target_reg[:, 2:] = xyxy2xywhn(
                pseudo_box_reg[:, 0:4],
                w=unlabel_imgs_weak_aug.size()[2],
                h=unlabel_imgs_weak_aug.size()[3],
            )
            unlabel_targets_merge_reg = torch.cat(
                [unlabel_targets_merge_reg, unlabel_target_reg]
            )
            semi_label_items += n_box
        return (
            unlabel_class_one_hot,
            unlabel_targets_merge_cls,
            unlabel_targets_merge_reg,
        )

    def train_on_batch_semi(
        self,
        label_imgs,
        label_targets,
        unlabel_imgs_strong_aug,
        label_class_one_hot,
        unlabel_class_one_hot,
        unlabel_targets_merge_cls,
        unlabel_targets_merge_reg,
    ):
        Bl = label_imgs.size()[0]
        label_unlabel_imgs = torch.cat([label_imgs, unlabel_imgs_strong_aug])
        with amp.autocast(enabled=self.device):
            pred, pred_mls = self.student_model(label_unlabel_imgs)  # forward
            sup_pred = [p[:Bl] for p in pred]
            semi_pred = [p[Bl:] for p in pred]
            sup_pred_mls, semi_pred_mls = pred_mls[:Bl], pred_mls[Bl:]
            loss, loss_items = self.compute_loss(
                sup_pred,
                label_targets.to(self.device),
                sup_pred_mls,
                label_class_one_hot.to(self.device),
            )  # loss scaled by batch_size

            # pay attention: ignoring the regression term
            semi_loss_cls, semi_loss_items_cls = self.compute_semi_loss(
                semi_pred,
                unlabel_targets_merge_cls.to(self.device),
                semi_pred_mls,
                unlabel_class_one_hot.to(self.device),
                cls_only=True,
            )  # loss scaled by batch_size
            semi_loss_reg, semi_loss_items_reg = self.compute_semi_loss(
                semi_pred,
                unlabel_targets_merge_reg.to(self.device),
                semi_pred_mls,
                unlabel_class_one_hot.to(self.device),
                box_only=True,
            )  # loss scaled by batch_size
            semi_loss = semi_loss_cls + semi_loss_reg
            semi_loss_items = semi_loss_items_cls + semi_loss_items_reg

            if self.opt.quad:
                semi_loss *= 4.0
                loss *= 4
        return loss, semi_loss, semi_loss_items


    def plot_on_batch(self):
        # Log
        if self.rank in [-1, 0]:
            mean_loss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mloss_semi = (mloss_semi * i + semi_loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 9) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, *mloss_semi,label_targets.shape[0], semi_label_items))
            callbacks.run('on_train_batch_end', ni, model, label_imgs, label_targets, label_paths, plots, opt.sync_bn)
        # end batch ------------------------------------------------------------------------------------------------
        if RANK in [-1, 0] and i==len(pbar)-1:
            return