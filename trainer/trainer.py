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
from utils.datasets import create_dataloader

from utils.general import check_dataset, check_file, check_img_size, colorstr, init_seeds, labels_to_class_weights, one_cycle
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images
from utils.torch_utils import ModelEMA, torch_distributed_zero_first
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, opt, device, LOCAL_RANK, RANK, WORLD_SIZE) -> None:
        self.opt = opt
        self.device = device

        # Set up env
        self.resume()
        self.check_DDP_mode()
        self.hyp = self.load_hyperparameter()
        logger.info(opt)


        # Set up logging and saving
        self.save_dir, self.epochs, self.batch_size, self.total_batch_size, self.weights, self.rank, self.freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze
        self.create_save_folder()
        self.wandb_logger, self.data_dict = self.log_wandb()
        self.nc = int(self.data_dict['nc'])  # number of classes
        self.names = self.data_dict['names']  # class names
        self.nominal_batch_size = 64  # nominal batch size
        accumulate = max(round(self.nominal_batch_size / self.total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.total_batch_size * accumulate / self.nominal_batch_size  # scale weight_decay
        logger.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")

        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, opt.data)  # check
       
        # Load Models
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()


        # EMA
        self.ema = ModelEMA(self.model) if   self.rank in [-1, 0] else None

        self.dataset, self.dataloader, self.valloader, self.nl, self.nb, self.imgsz, self.imgsz_test = self.load_dataset()

        # Model parameters
        self.hyp['box'] *= 3. / self.nl  # scale to layers
        self.hyp['cls'] *= self.nc / 80. * 3. / self.nl  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3. / self.nl  # scale to image size and layers
        self.hyp['label_smoothing'] = opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(device) * self.nc  # attach class weights
        self.model.names = self.names

      

    def resume(self):
        """Resume logger and config
        """
        
        wandb_run = check_wandb_resume(self.opt)
        if self.opt.resume and not wandb_run:  # resume an interrupted run
            pass
        else:
            # self.opt.hyp = self.opt.hyp or ('hyp.finetune.yaml' if self.opt.weights else 'hyp.scratch.yaml')
            self.opt.data, self.opt.cfg, self.opt.hyp = check_file(self.opt.data), check_file(self.opt.cfg), check_file(self.opt.hyp)  # check files
            assert len(self.opt.cfg) or len(self.opt.weights), 'either --cfg or --weights must be specified'
            self.self.opt.img_size.extend([self.opt.img_size[-1]] * (2 - len(self.opt.img_size)))  # extend to 2 sizes (train, test)
            self.opt.name = 'evolve' if self.opt.evolve else self.opt.name

    def check_DDP_mode(self):
        """DDP mode
        """
        self.opt.total_batch_size =self.opt.batch_size
        if self.opt.local_rank != -1:
            assert torch.cuda.device_count() >self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            device = torch.device('cuda',self.opt.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert self.opt.batch_size %self.opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            self.opt.batch_size =self.opt.total_batch_size //self.opt.world_size

    def load_hyperparameter(self):
        """Hyperparameters
        """
        with open(self.opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
        logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        return hyp

    def create_save_folder(self):
        # Directories
        self.wdir = wdir = self.save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last = wdir / 'last.pt'
        self.best = wdir / 'best.pt'
        self.results_file = self.save_dir / 'results.txt'
        # Save run settings
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(self.save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)
         # Configure
        init_seeds(2 + self.rank)
        with open(self.opt.data) as f:
            self.data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict


    def log_wandb(self):
        # Logging- Doing this before checking the dataset. Might update data_dict
        loggers = {'wandb': None}  # loggers dict
        if self.rank in [-1, 0]:
            self.opt.hyp = self.hyp  # add hyperparameters
            run_id = torch.load(self.weights, map_location=self.device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
            wandb_logger = WandbLogger(self.opt, Path(self.opt.save_dir).stem, run_id, self.data_dict)
            loggers['wandb'] = wandb_logger.wandb
            data_dict = wandb_logger.data_dict
            if wandb_logger.wandb:
                weights, epochs, hyp = self.opt.weights, self.opt.epochs, self.opt.hyp  # WandbLogger might update weights, epochs if resuming
        return wandb_logger, data_dict
    
    def load_model(self):
        model = Model(self.opt.cfg, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
        # Freeze
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False
        
        return model

    def load_dataset(self):
        opt = self.opt
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)  # check
        train_path = self.data_dict['train']
        test_path = self.data_dict['val']


        # Image sizes
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

        # Trainloader

        dataloader, dataset = create_dataloader(train_path, imgsz, self.batch_size, gs, opt,
                                                hyp=self.hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=self.rank,
                                                world_size=opt.world_size, workers=opt.workers,
                                                image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, self.nc, opt.data, self.nc - 1)
        if self.rank in [-1, 0]:
            valloader = create_dataloader(test_path, imgsz_test, self.batch_size * 2, gs, opt,  # testloader
                                        hyp=self.hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                        world_size=opt.world_size, workers=opt.workers,
                                        pad=0.5, prefix=colorstr('val: '))[0]
        return dataset, dataloader, valloader, nl, nb, imgsz, imgsz_test

    def load_optimizer(self):
        
        # optimizer parameter groups
        pg0, pg1, pg2 = [], [], [] 
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):           
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):           
                    pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):           
                    pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):           
                    pg0.append(v.imo.implicit)
                else:
                    for iv in v.imo:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):           
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)
            if hasattr(v, 'attn'):
                if hasattr(v.attn, 'logit_scale'):   
                    pg0.append(v.attn.logit_scale)
                if hasattr(v.attn, 'q_bias'):   
                    pg0.append(v.attn.q_bias)
                if hasattr(v.attn, 'v_bias'):  
                    pg0.append(v.attn.v_bias)
                if hasattr(v.attn, 'relative_position_bias_table'):  
                    pg0.append(v.attn.relative_position_bias_table)
            if hasattr(v, 'rbr_dense'):
                if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                    pg0.append(v.rbr_dense.weight_rbr_origin)
                if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                    pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                    pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                    pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                    pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(v.rbr_dense, 'vector'):   
                    pg0.append(v.rbr_dense.vector)

        if self.opt.adam:
            optimizer = optim.Adam(self.pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(self.pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        return optimizer

    def load_scheduler(self):
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        if self.opt.linear_lr:
            lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - self.hyp['lrf']) + hyp['lrf']  # linear
        else:
            lf = one_cycle(1, self.hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        return scheduler
    
    def train(self):
        # Start training
        model = self.model
        t0 = time.time()
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        self.maps = np.zeros(self.nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        start_epoch, best_fitness = 0, 0.0
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.device)
        compute_loss_ota = ComputeLossOTA(model)  # init loss class
        compute_loss = ComputeLoss(model)  # init loss class
        torch.save(model, self.wdir / 'init.pt')


        for epoch in range(start_epoch, self.epochs): 
            self.train_on_epoch(epoch, model, compute_loss_ota, compute_loss)


    def train_on_epoch(self, epoch, model, compute_loss_ota, compute_loss):
        """train 1 epochs

        Args:
            epoch (_type_): _description_
            model (_type_): _description_
            compute_loss_ota (_type_): _description_
            compute_loss (_type_): _description_
        """
        model.train()

        if self.rank != -1:
            self.dataloader.sampler.set_epoch(epoch)
        mloss = torch.zeros(4, device=self.device)  # mean losses
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = enumerate(self.dataloader)
        if self.rank in [-1, 0]:
            pbar = tqdm(pbar, total=self.nb)  
        self.optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar: 

            ni = i + self.nb * epoch
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            # Warm up --------------------------------
            if ni <= self.nw:
                xi = [0, self.nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, self.nominal_batch_size / self.total_batch_size]).round())
                self.warm_up(ni, xi)
            
            # Train on batch -----------------------------
            loss_items = self.train_on_batch(model, imgs, targets, compute_loss_ota, compute_loss)
            # Optimize
            if ni % accumulate == 0:
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(model)

            self.plot_on_batch( epoch, i, loss_items,pbar,  imgs, targets, paths)
        # Scheduler
        lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
        self.scheduler.step()

        self.end_epoch(epoch, compute_loss_ota, compute_loss)


    
    def warm_up(self, ni, xi):
        # Warmup
        for j, x in enumerate(self.optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

    def train_on_batch(self, model, imgs, targets,compute_loss_ota, compute_loss):
        
        # Forward
        with amp.autocast(enabled=self.device):
            pred = model(imgs)  # forward
            if 'loss_ota' not in self.hyp or self.hyp['loss_ota'] == 1:
                loss, loss_items = compute_loss_ota(pred, targets.to(self.device), imgs)  # loss scaled by batch_size
            else:
                loss, loss_items = compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
          
            if self.opt.quad:
                loss *= 4.

        # Backward
        self.scaler.scale(loss).backward()
        return loss_items

      
    def plot_on_batch(self, epoch, i, loss_items, pbar, imgs, targets, paths):
        # Print
        if self.rank in [-1, 0]:
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, self.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if i < 10:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
             
            elif i == 10 and self.wandb_logger.wandb:
                self.wandb_logger.log({"Mosaics": [self.wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                self.save_dir.glob('train*.jpg') if x.exists()]})

    def end_epoch(self, epoch, compute_loss_ota, compute_loss):
        # Scheduler
        lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
        self.scheduler.step()

        # DDP process 0 or single-GPU
        if self.rank in [-1, 0]:
            # mAP
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == self.epochs
            if not self.opt.notest or final_epoch:  # Calculate mAP
                self.wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(self.data_dict,
                                                 batch_size=self.batch_size * 2,
                                                 imgsz=self.imgsz_test,
                                                 model=self.ema.ema,
                                                 single_cls=self.opt.single_cls,
                                                 dataloader=self.valloader,
                                                 save_dir=self.save_dir,
                                                 verbose=self.nc < 50 and final_epoch,
                                                 plots=final_epoch,
                                                 wandb_logger=self.wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=False,
                                                 v5_metric=self.opt.v5_metric)

            # Write
            with open(self.results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(self.opt.name) and self.opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if self.wandb_logger.wandb:
                    self.wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            self.wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not self.opt.nosave) or (final_epoch and not self.opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id': self.wandb_logger.wandb_run.id if self.wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, self.wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, self.wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, self.wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (self.epochs-5):
                    torch.save(ckpt, self.wdir / 'epoch_{:03d}.pt'.format(epoch))
                if self.wandb_logger.wandb:
                    if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        self.wandb_logger.log_model(
                            self.last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt