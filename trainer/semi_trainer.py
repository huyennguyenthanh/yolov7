from copy import deepcopy
from formatter import test
import json
import os
from threading import Thread
from time import time
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
from utils.datasets_semi import create_dataloader_semi

# from utils.datasets import create_dataloader
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
from utils.loss.compute_loss import ComputeLoss
from utils.loss.compute_loss_ota import ComputeLossOTA
from utils.datasets import create_dataloader
from utils.loss.compute_loss_semi import ComputeLossOTASemi
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
        self.model = self.load_model()  # student_model
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
        ) = self.load_dataset(type="train", num_images=200, augment=False, mosaic=True)
        (
            self.unlabeled_dataset,
            self.unlabeled_dataloader,
            self.unlabeled_num_batches,
        ) = self.load_dataset(
            type="unlabel", num_images=2000, augment=True, mosaic=True
        )
        (
            self.val_dataset,
            self.val_dataloader,
            self.val_num_batches,
        ) = self.load_val_dataset()
        self.num_batches = max(self.labeled_num_batches, self.unlabeled_num_batches)
        # EMA
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None

        # Model parameters
        self.hyp["box"] *= 3.0 / self.num_layers  # scale to layers
        self.hyp["cls"] *= (
            self.nc / 80.0 * 3.0 / self.num_layers
        )  # scale to classes and layers
        self.hyp["obj"] *= (
            (self.imgsz / 640) ** 2 * 3.0 / self.num_layers
        )  # scale to image size and layers
        self.hyp["label_smoothing"] = opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

        self.model.class_weights = (
            labels_to_class_weights(self.labeled_dataset.labels, self.nc).to(device)
            * self.nc
        )  # attach class weights
        self.model.names = self.names

        self.time_dict = {
            "plot_image_on_batch": 0,
            "warm_up": 0,
            "extract_data": 0,
            "train_on_batch_supervised": 0,
            "update_ema": 0,
            "predict_pseudo_label": 0,
            "train_on_batch_semi": 0,
            "backward": 0,
            "optimizer": 0,
            "scheduler": 0,
        }
        with open("time.json", "w") as f:
            json.dump(self.time_dict, f)
            f.write(os.linesep)

    def load_dataset(
        self, type="train", num_images=10000000, augment=False, mosaic=False
    ):
        opt = self.opt
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)  # check
        path = self.data_dict[type]

        # Trainloader
        dataloader, dataset = create_dataloader_semi(
            path,
            self.imgsz,
            opt.batch_size,
            self.gs,
            opt,
            hyp=self.hyp,
            augment=augment,
            cache=opt.cache_images,
            rect=opt.rect,
            rank=self.rank,
            world_size=opt.world_size,
            workers=opt.workers,
            image_weights=opt.image_weights,
            quad=opt.quad,
            prefix=colorstr("train: "),
            mosaic=mosaic,
            num_images=num_images,
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
            dataloader, dataset = create_dataloader(
                self.data_dict["val"],
                self.imgsz_test,
                opt.batch_size * 4,
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
                num_images=200,
            )
        nb = len(dataloader)
        return dataset, dataloader, nb

    def train(self):

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
        self.compute_loss = ComputeLoss(self.model)
        self.compute_loss_ota = ComputeLossOTA(self.model)
        self.compute_loss_semi = ComputeLossOTASemi(self.model)

        for epoch in range(start_epoch, self.opt.epochs):
            self.train_on_epoch_semi(epoch)
        self.end_training()

    def train_on_epoch_semi(self, epoch):
        self.model.train()
        mean_loss = torch.zeros(4, device=self.device)  # mean losses
        mean_loss_semi = torch.zeros(3, device=self.device)  # mean losses

        if self.rank != -1:
            self.labeled_dataloader.sampler.set_epoch(epoch)
            self.unlabeled_dataloader.sampler.set_epoch(epoch)
            self.labeled_dataloader.set_length(
                max(len(self.labeled_dataloader), len(self.unlabeled_dataloader))
            )
            self.unlabeled_dataloader.set_length(
                max(len(self.labeled_dataloader), len(self.unlabeled_dataloader))
            )
        pbar = enumerate(zip(self.labeled_dataloader, self.unlabeled_dataloader))

        logger.info(
            ("\n" + "%10s" * 9)
            % (
                "Epoch",
                "gpu_mem",
                "box",
                "obj",
                "cls",
                "total",
                "semi_obj",
                "semi_cls",
                "semi_reg",
            )
        )
        pbar = tqdm(pbar, total=self.num_batches)  # progress bar
        self.optimizer.zero_grad()

        for i, data in pbar:

            ni = i + self.num_batches * epoch  # num iteration
            start = time()
            if i < 5 and epoch < 3:
                self.plot_image_on_batch(i, epoch, data)
            self.time_dict["plot_image_on_batch"] = time() - start
            # Warm up --------------------------------
            start = time()
            if ni <= self.num_warm_up_iters:
                self.warm_up(epoch, ni)
            self.time_dict["warm_up"] = time() - start

            mean_loss, mean_loss_semi = self.train_on_batch(
                epoch, i, ni, data, mean_loss, mean_loss_semi
            )

            self.plot_on_batch(pbar, epoch, mean_loss, mean_loss_semi)

            if epoch < self.hyp["burn_up_epoch"] and i > self.labeled_num_batches:
                print(epoch, self.hyp["burn_up_epoch"], i, self.labeled_num_batches)

                break

            if i == len(pbar) - 1:
                print(i, len(pbar))
                break
            if i % 10 == 0:
                with open("time.json", "a") as f:
                    json.dump(self.time_dict, f)
                    f.write(os.linesep)

        start = time()
        self.scheduler.step()
        self.time_dict["scheduler"] = time() - start
        self.end_epoch(epoch, mean_loss, mean_loss_semi)

    def plot_image_on_batch(self, i, epoch, data, paths=None):

        (label_imgs, label_targets, label_class_one_hot), (
            unlabel_imgs,
            unlabel_targets,
            unlabel_class_one_hot,
        ) = data
        label_imgs_weak_aug, label_imgs_strong_aug = label_imgs
        unlabel_imgs_weak_aug, unlabel_imgs_strong_aug = unlabel_imgs

        f = (
            self.save_dict["save_dir"] / f"train_epoch_{epoch}_batch_{i}_label.jpg"
        )  # filename
        Thread(
            target=plot_images,
            args=(label_imgs_weak_aug, label_targets, paths, f),
            daemon=True,
        ).start()
        f = (
            self.save_dict["save_dir"]
            / f"train_epoch_{epoch}_batch_{i}_unlabel_weak.jpg"
        )  # filename
        Thread(
            target=plot_images,
            args=(unlabel_imgs_weak_aug, unlabel_targets, paths, f),
            daemon=True,
        ).start()
        f = (
            self.save_dict["save_dir"]
            / f"train_epoch_{epoch}_batch_{i}_unlabel_strong.jpg"
        )  # filename
        Thread(
            target=plot_images,
            args=(unlabel_imgs_strong_aug, unlabel_targets, paths, f),
            daemon=True,
        ).start()

    def train_on_batch(self, epoch, i, ni, data, mean_loss, mean_loss_semi):
        """_summary_

        Args:
            i (_type_): _description_
            ni (_type_): _description_
            data (_type_): tuple
                label_targets: torch.Size([bs, 6])
                label_class_one_hot :torch.Size([bs, 1])

                label_imgs_weak_aug: torch.Size([bs, 3, 640, 640])
                unlabel_imgs_weak_aug: torch.Size([bs, 3, 640, 640])
            mean_loss (): torch.Size([4])
            mean_loss_semi (): torch.Size([3])

        Returns:
            _type_: _description_
        """
        start = time()
        (label_imgs, label_targets, label_class_one_hot), (
            unlabel_imgs,
            unlabel_targets,
            unlabel_class_one_hot,
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
        # label_class_one_hot = torch.cat(
        #     [label_class_one_hot, label_class_one_hot.detach().clone()], 0
        # )
        label_targets_strong = label_targets.clone().detach()

        if label_targets.size()[0] == 0 or label_targets_strong.size()[0] == 0:
            print(label_imgs_weak_aug.size())
            return
        label_targets_strong[:, 0] += label_targets[-1, 0] + 1
        label_targets = torch.cat([label_targets, label_targets_strong], 0)
        self.time_dict["extract_data"] = time() - start
        semi_loss = 0.0
        semi_loss_items = torch.zeros(3, device=self.device)
        start = time()
        if (
            i < self.labeled_num_batches and epoch < self.hyp["burn_up_epoch"]
        ) or ni < self.hyp["burn_up_step"]:

            loss, loss_items = self.train_on_batch_supervised(
                ni, label_imgs, label_targets
            )
            self.time_dict["train_on_batch_supervised"] = time() - start

        else:
            self.time_dict["train_on_batch_supervised"] = 0

            self.update_ema(ni)
            self.time_dict["update_ema"] = time() - start
            start = time()

            (
                unlabel_targets_merge_cls,
                unlabel_targets_merge_reg,
            ) = self.predict_pseudo_label(unlabel_imgs_weak_aug)
            self.time_dict["predict_pseudo_label"] = time() - start
            start = time()
            loss, loss_items, semi_loss, semi_loss_items = self.train_on_batch_semi(
                label_imgs,
                label_targets,
                unlabel_imgs_strong_aug,
                unlabel_targets_merge_cls,
                unlabel_targets_merge_reg,
            )
            self.time_dict["train_on_batch_semi"] = time() - start

        loss = loss + semi_loss * self.hyp["semi_loss_weight"]
        semi_loss_items *= self.hyp["semi_loss_weight"]
        mean_loss = (mean_loss * i + loss_items) / (i + 1)
        mean_loss_semi = (mean_loss_semi * i + semi_loss_items) / (i + 1)
        # Backward
        start = time()
        self.scaler.scale(loss).backward()
        self.time_dict["backward"] = time() - start

        # Optimize
        start = time()
        if ni % self.accumulate == 0:

            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema is not None:
                self.ema.update(self.model)
        self.time_dict["optimizer"] = time() - start

        return mean_loss, mean_loss_semi

    def train_on_batch_supervised(self, ni, label_imgs, label_targets):

        with amp.autocast(enabled=self.device != "cpu"):
            pred = self.model(label_imgs)  # forward
            loss, loss_items = self.compute_loss_ota(
                pred, label_targets.to(self.device), label_imgs
            )  # loss scaled by batch_size
            if self.opt.quad:
                loss *= 4.0
        return loss, loss_items

    def update_ema(self, ni):
        if ni == self.hyp["burn_up_step"]:
            self.teacher_model = _update_teacher_model(
                self.model if self.ema is None else self.ema.ema,
                self.teacher_model,
                keep_rate=0.0,
            )
        elif (ni - self.hyp["burn_up_step"]) % self.hyp["teacher_update_iter"] == 0:
            self.teacher_model = _update_teacher_model(
                self.model if self.ema is None else self.ema.ema,
                self.teacher_model,
                keep_rate=self.hyp["ema_keep_rate"],
            )

    def predict_pseudo_label(self, unlabel_imgs_weak_aug):

        # Predict
        self.teacher_model.eval()
        with torch.no_grad():
            out = self.teacher_model(unlabel_imgs_weak_aug, augment=True)[0]

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

        unlabel_targets_merge_reg = torch.zeros(0, 6).to(self.device)
        unlabel_targets_merge_cls = torch.zeros(0, 6).to(self.device)
        # Filter
        for batch_ind, (pseudo_box_reg, pseudo_box_cls) in enumerate(
            zip(pseudo_boxes_reg, pseudo_boxes_cls)
        ):

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
        return (
            unlabel_targets_merge_cls,
            unlabel_targets_merge_reg,
        )

    def train_on_batch_semi(
        self,
        label_imgs,
        label_targets,
        unlabel_imgs_strong_aug,
        unlabel_targets_merge_cls,
        unlabel_targets_merge_reg,
    ):
        Bl = label_imgs.size()[0]
        label_unlabel_imgs = torch.cat([label_imgs, unlabel_imgs_strong_aug])
        with amp.autocast(enabled=self.device != "cpu"):
            pred = self.model(label_unlabel_imgs)  # forward
            sup_pred = [p[:Bl] for p in pred]
            semi_pred = [p[Bl:] for p in pred]

            loss, loss_items = self.compute_loss_ota(
                sup_pred, label_targets.to(self.device), label_imgs
            )
            semi_loss_cls, semi_loss_items_cls = self.compute_loss_semi(
                semi_pred,
                unlabel_targets_merge_cls.to(self.device),
                unlabel_imgs_strong_aug,
                cls_only=True,
            )  # loss scaled by batch_size
            semi_loss_reg, semi_loss_items_reg = self.compute_loss_semi(
                semi_pred,
                unlabel_targets_merge_reg.to(self.device),
                unlabel_imgs_strong_aug,
                bbox_only=True,
            )  # loss scaled by batch_size
            semi_loss = semi_loss_cls + semi_loss_reg
            semi_loss_items = semi_loss_items_cls + semi_loss_items_reg

            if self.opt.quad:
                semi_loss *= 4.0
                loss *= 4
        return loss, loss_items, semi_loss, semi_loss_items

    def plot_on_batch(self, pbar, epoch, mean_loss, mean_loss_semi):

        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
        pbar.set_description(
            ("%10s" * 2 + "%10.4g" * 7)
            % (f"{epoch}/{self.opt.epochs - 1}", mem, *mean_loss, *mean_loss_semi)
        )

    def end_epoch(self, epoch, mean_loss, mean_loss_semi):

        if self.ema is not None:
            self.ema.update_attr(
                self.model,
                include=["yaml", "nc", "hyp", "names", "stride", "class_weights"],
            )

        final_epoch = epoch + 1 == self.opt.epochs
        if not self.opt.notest or final_epoch:  # Calculate mAP
            if epoch < self.hyp["burn_up_epoch"]:
                model = deepcopy(self.model)
            else:
                model = deepcopy(self.teacher_model)
            # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist())
            results, maps, times = test.test(
                self.data_dict,
                batch_size=self.opt.batch_size * 4,
                imgsz=self.imgsz_test,
                model=model,
                single_cls=self.opt.single_cls,
                dataloader=self.val_dataloader,
                save_dir=self.save_dict["save_dir"],
                verbose=self.nc < 50 and final_epoch,
                plots=final_epoch,
                wandb_logger=self.wandb_logger,
                compute_loss=self.compute_loss,
                is_coco=False,
                v5_metric=self.opt.v5_metric,
            )

        # Update best mAP
        fi = fitness(
            np.array(results).reshape(1, -1)
        )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > self.best_fitness:
            self.best_fitness = fi

        # Write
        mean_loss = mean_loss.detach().cpu().numpy().tolist()
        s = ("%10s" * 2 + "%10.4g" * 3) % (
            "%g/%g" % (epoch, self.opt.epochs - 1),
            *mean_loss,
        )
        with open(self.save_dict["results_file"], "a") as f:
            f.write(s + "%10.4g" * 7 % results + "\n")  # append metrics, val_loss
        # Log
        tags = [
            "train/box_loss",
            "train/obj_loss",
            "train/cls_loss",  # train loss
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",  # val loss
        ]  # params
        for x, tag in zip(list(mean_loss[:-1]) + list(results), tags):
            if self.wandb_logger.wandb:
                self.wandb_logger.log({tag: x})  # W&B

        # Save model
        if (not self.opt.nosave) or (final_epoch and not self.opt.evolve):  # if save
            ckpt = {
                "epoch": epoch,
                "best_fitness": self.best_fitness,
                "model_student": deepcopy(self.model).half(),
                "model": deepcopy(self.teacher_model).half(),
                "optimizer": self.optimizer.state_dict(),
                "wandb_id": self.wandb_logger.wandb_run.id
                if self.wandb_logger.wandb
                else None,
            }

            # Save last, best and delete
            torch.save(ckpt, self.save_dict["last"])
            if self.best_fitness == fi:
                torch.save(ckpt, self.save_dict["best"])
            elif ((epoch + 1) % 25) == 0:
                torch.save(
                    ckpt, self.save_dict["wdir"] / "epoch_{:03d}.pt".format(epoch)
                )
            elif epoch >= (self.opt.epochs - 5):
                torch.save(
                    ckpt, self.save_dict["wdir"] / "epoch_{:03d}.pt".format(epoch)
                )
            if self.wandb_logger.wandb:
                if (
                    (epoch + 1) % self.opt.save_period == 0 and not final_epoch
                ) and self.opt.save_period != -1:
                    self.wandb_logger.log_model(
                        self.last.parent,
                        self.opt,
                        epoch,
                        fi,
                        best_model=self.best_fitness == fi,
                    )
            del ckpt
