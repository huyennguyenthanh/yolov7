import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import traceback
import json

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.datasets_semi import  create_dataloader_label, create_dataloader_semi
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, non_max_suppression, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.semi_psuedo_label_process import non_max_suppression_pseudo_decouple_multi_view
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.loss_semi import ComputeLossOTASemi, ComputeLossSemi
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.semi_psuedo_label_process import convert_output_to_label_2, convert_to_eval_output
from utils.torch_utils import ModelEMA, _update_teacher_model, select_device, torch_distributed_zero_first, is_parallel, update_teacher_model_mpl
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from utils.load_model import load_model
logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    time_file = save_dir / 'time.json'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = None #torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

   
    
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    unlabel_path = data_dict['unlabel']
    val_path = data_dict['val']
    test_path = data_dict['test']

 
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    print("Accumulate: ", accumulate)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    print("Load student")
    model, s_optimizer, lf, s_scheduler = load_model(opt, hyp, nc=data_dict['nc'], weights=weights, device=device)
    print("Load teacher")
    model_teacher, t_optimizer, lf, t_scheduler = load_model(opt, hyp, nc=data_dict['nc'], weights=weights, device=device, model_teacher=True)

   
    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None
    if weights.endswith('.pt'):
        ckpt = torch.load(weights, map_location=device) 
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            # ema.updates = ckpt['updates']

    # Resume
    start_epoch, best_fitness = 0, 0.0
   

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model_teacher = torch.nn.DataParallel(model_teacher)

  

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    train_label_loader, dataset_label = create_dataloader_label(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, mosaic=True, prefix=colorstr('train: '))
    hyp["mosaic"] = 0.4
    hyp["mixup"] = 0.
    train_unlabel_loader, dataset_unlabel = create_dataloader_semi(unlabel_path, imgsz, batch_size,
                                                                   gs, opt,
                                                                   hyp=hyp, augment=False, cache=opt.cache_images,
                                                                   rect=opt.rect, rank=rank,  world_size=opt.world_size,
                                                                   workers=opt.workers, image_weights=opt.image_weights,
                                                                   quad=opt.quad, mosaic=True,
                                                                   prefix=colorstr('train: '))
    mlc = int(np.concatenate(dataset_label.labels, 0)[:,0].max())  
    print(f"Len label dataset: {dataset_label.__len__()} - Len unlabel dataset: {dataset_unlabel.__len__()}")
    nb = max(len(train_label_loader),len(train_unlabel_loader) ) # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        valloader = create_dataloader(val_path, imgsz_test, batch_size*4, gs, opt,  # valloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        testloader = create_dataloader(test_path, imgsz_test, batch_size*4, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]


    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset_label.labels, nc).to(device) * nc  # attach class weights
    model.names = names


    model_teacher.nc = nc  # attach number of classes to model
    model_teacher.hyp = hyp  # attach hyperparameters to model
    model_teacher.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model_teacher.class_weights = labels_to_class_weights(dataset_label.labels, nc).to(device) * nc  # attach class weights
    model_teacher.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    s_scheduler.last_epoch = start_epoch - 1  # do not move
    t_scaler = amp.GradScaler(enabled=cuda)
    s_scaler = amp.GradScaler(enabled=cuda)

    compute_loss_ota = ComputeLossOTASemi(model) 
    compute_loss = ComputeLossSemi(model) 

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {train_label_loader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')

    bbox_threshold = hyp['bbox_threshold']
    cls_threshold = hyp['cls_threshold']
    time_dict = {
        "epoch" : 0,
        "i": 0,
        "load_data": 0.,
        "train_on_batch_supervised_foward": 0.,
        "train_on_batch_supervised_loss": 0,
        "update_teacher": 0,
        "predict_pseudo_label": 0,
        "predict_pseudo_label_filter": 0,
        "train_on_batch_semi_foward": 0,
        "train_on_batch_semi_loss": 0,
        "loss_backward":0,
        "optimizer":0,
        "evaluate":0,
        "scheduler":0,
        "all": 0
    }
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        start_all = time.time()
       
        time_dict["epoch"] = epoch
     
        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset_label.labels, nc=nc, class_weights=cw)  # image weights
                dataset_label.indices = random.choices(range(dataset_label.n), weights=iw, k=dataset_label.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset_label.indices) if rank == 0 else torch.zeros(dataset_label.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset_label.indices = indices.cpu().numpy()

        semi_label_items = torch.zeros(1, device=device)
        s_loss_l_old = s_loss = s_loss_l_new = s_loss_u = s_loss_mpl = torch.zeros(1, device=device)

        if rank != -1:
            train_label_loader.sampler.set_epoch(epoch)
            train_unlabel_loader.sampler.set_epoch(epoch)
            train_label_loader.set_length(max(len(train_label_loader),len(train_unlabel_loader)))
            train_unlabel_loader.set_length(max(len(train_label_loader),len(train_unlabel_loader)))

        pbar = enumerate(zip(train_label_loader,train_unlabel_loader))

        logger.info(('\n' + '%13s' * 9) % ('Epoch', 'gpu_mem',  's_loss_l_old', 's_loss_u', 's_loss', 's_loss_l_new', 's_loss_mpl','labels', 'semi_labels'))
        with open(results_file, 'a') as f:
            f.write( '%13s' * 9 % ('Epoch', 'gpu_mem',  's_loss_l_old', 's_loss_u', 's_loss', 's_loss_l_new','s_loss_mpl', 'labels', 'semi_labels') \
                + '%10s' * 7 % ('Val/Precision', 'Val/Recall', 'Val/mAP50', 'Val/mAP50:95', 'loss_bbox', 'loss_obj', 'loss_cls') \
                + '%10s' * 7 % ('Val/Precision', 'Val/Recall', 'Val/mAP50', 'Val/mAP50:95', 'loss_bbox', 'loss_obj', 'loss_cls'))
            f.write('\n')

        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        
        
        s_optimizer.zero_grad()
        t_optimizer.zero_grad()
        # if not epoch <= hyp["supervised_epoch"]:
        #     bbox_threshold = max(0.5,bbox_threshold - 0.05)
        #     cls_threshold = min(0.25, cls_threshold + 0.1)
        for i, data in pbar:  # batch -------------------------------------------------------------
            time_dict["i"] = i
            start = time.time()
            semi_label_items = torch.zeros(1, device=device)
            (label_imgs, label_targets, label_paths, label_shapes), (
                unlabel_imgs, unlabel_targets, unlabel_paths, unlabel_shapes) = data
            label_imgs_weak_aug, label_imgs_strong_aug = label_imgs
            unlabel_imgs_weak_aug, unlabel_imgs_strong_aug = unlabel_imgs
        
            if (i % round(nb/5) == 0) and epoch == 0:
                f =  save_dir / f"train_epoch_{epoch}_batch_{i}_label_weak.jpg"
                Thread(
                    target=plot_images,
                    args=(label_imgs_weak_aug, label_targets, None, f),
                    daemon=True,
                ).start()
                f =  save_dir / f"train_epoch_{epoch}_batch_{i}_label_strong.jpg"
                Thread(
                    target=plot_images,
                    args=(label_imgs_strong_aug, label_targets, None, f),
                    daemon=True,
                ).start()
            
           

            label_imgs_weak_aug = label_imgs_weak_aug.to(device, non_blocking=True).float() / 255
            label_imgs_strong_aug = label_imgs_strong_aug.to(device, non_blocking=True).float() / 255
            unlabel_imgs_weak_aug = unlabel_imgs_weak_aug.to(device, non_blocking=True).float()  / 255
            unlabel_imgs_strong_aug = unlabel_imgs_strong_aug.to(device, non_blocking=True).float()  / 255
            label_targets=label_targets.to(device, non_blocking=True)
            unlabel_targets = unlabel_targets.to(device, non_blocking=True)
        
            time_dict["load_data"] = time.time() - start
            start = time.time()
    
            
            ni = i + nb * epoch  

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(s_optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(label_imgs_weak_aug.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in label_imgs_weak_aug.shape[2:]]  # new shape (stretched to gs-multiple)
                    # imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    label_imgs_weak_aug = F.interpolate(label_imgs_weak_aug, size=ns, mode='bilinear', align_corners=False)
                    label_imgs_strong_aug = F.interpolate(label_imgs_strong_aug, size=ns, mode='bilinear', align_corners=False)
                    unlabel_imgs_weak_aug = F.interpolate(unlabel_imgs_weak_aug, size=ns, mode='bilinear', align_corners=False)
                    unlabel_imgs_strong_aug = F.interpolate(unlabel_imgs_strong_aug, size=ns, mode='bilinear', align_corners=False)


       
          
            # Forward
           

            if epoch < hyp["supervised_epoch"]:
                """1. Supervised part
                """
    
                label_imgs=torch.cat([label_imgs_weak_aug, label_imgs_strong_aug],0) # [bsx2, 3, 640, 640]
                label_targets_strong = label_targets.clone().detach()
                label_targets_strong[:,0] += (label_targets[-1,0]+1)
                label_targets = torch.cat([label_targets,label_targets_strong],0)
              
                with amp.autocast(enabled=cuda):
                    pred = model(label_imgs)  # forward

                    time_dict["train_on_batch_supervised_foward"] = time.time() - start
                    start = time.time()
            
                    if hyp['loss_ota'] == 1:
                        s_loss, s_loss_items = compute_loss_ota(pred, label_targets.to(device), label_imgs)  # loss scaled by batch_size
                    else:
                        s_loss, s_loss_items = compute_loss(pred, label_targets.to(device))  # loss scaled by batch_size
                    
                    time_dict["train_on_batch_supervised_loss"] = time.time() - start
                    start = time.time()
                    
            
                    if rank != -1:
                        loss *= opt.world_size  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.
                
                s_scaler.scale(s_loss).backward()
            
                s_scaler.step(s_optimizer)  # optimizer.step
                s_scaler.update()
                s_optimizer.zero_grad()
                if ema:
                    ema.update(model)

            else:
                """Semi supervised part
                """
              
                if i == 0 and epoch == hyp["supervised_epoch"]:
                    print(f"Iteration {ni} - Epoch{epoch}: Update teacher") 
                    _update_teacher_model(model if ema is None else ema.ema, model_teacher, word_size=opt.world_size, keep_rate=0.)
                elif  i % hyp['teacher_update_iter'] == 0:
                    if i == 0:
                        print(f"Iteration {ni} - Epoch{epoch}: Update teacher")
                    _update_teacher_model(model if ema is None else ema.ema, model_teacher, word_size=opt.world_size, keep_rate=hyp['ema_keep_rate'])
                
                time_dict["update_teacher"] = time.time() - start
                start = time.time()
             
                try: 
                    """Predict pseudo label
                    """
                    model_teacher.eval()
                    with torch.no_grad():
                      
                        out = model_teacher(unlabel_imgs_weak_aug)
                        time_dict["predict_pseudo_label"] = time.time() - start
                        start = time.time()


                        pred = non_max_suppression(out[0], conf_thres=cls_threshold, iou_thres=bbox_threshold)
                        pseudo_label = convert_output_to_label_2(unlabel_imgs_weak_aug, pred, conf=True, device=device)
                     
                        semi_label_items = pseudo_label.shape[0]

                        if (i % round(nb/5) == 0):
                            f =  save_dir / f"train_epoch_{epoch}_batch_{i}_pseudo_label.jpg"
                            plot_images(unlabel_imgs_weak_aug, unlabel_targets, None, f,conf_thres=0.0, predictions=pseudo_label)
                    

                          
                        pseudo_label = pseudo_label[:,:6]
                        time_dict["predict_pseudo_label_filter"] = time.time() - start
                        start = time.time()


                    """Foward student
                    """

                    label_imgs = label_imgs_weak_aug  if i % 2 == 0 else label_imgs_strong_aug
                    Bl=label_imgs.size()[0]
                    label_and_unlabel_imgs=torch.cat([label_imgs,unlabel_imgs_strong_aug])
                 
                    with amp.autocast(enabled=cuda):
                        l = label_imgs.size()[0]
                        s_images = torch.cat((label_imgs, unlabel_imgs_strong_aug)) # [64,3,32,32]
                        s_pred = model(s_images) # [64,10]
                        s_pred_l = [p[:l] for p in s_pred]
                        s_pred_us = [p[l:] for p in s_pred]
                        del s_pred

                    
                        if hyp['loss_ota'] == 1:
                            s_loss_l_old, loss_items = compute_loss_ota(s_pred_l, label_targets.to(device), label_imgs)  
                        else:    
                            s_loss_l_old, loss_items = compute_loss(s_pred_l, label_targets.to(device))
                        

                        if hyp['loss_ota'] == 1:
                            s_loss_u, s_loss_items = compute_loss_ota(s_pred_us, pseudo_label.to(device), label_imgs)  # loss scaled by batch_size
                        else:
                            s_loss_u, s_loss_items = compute_loss(s_pred_us, pseudo_label.to(device))  # loss scaled by batch_size

                        s_loss = s_loss_l_old + hyp["semi_loss_weight"] * s_loss_u
                      
                        if rank != -1:
                            semi_loss *= opt.world_size  # gradient averaged between devices in DDP mode
                            loss*=opt.world_size
                        if opt.quad:
                            semi_loss *= 4.
                            loss *= 4
                        time_dict["train_on_batch_semi_loss"] = time.time() - start
                        start = time.time()

                 
                    '''3. loss backward
                    '''
                    s_scaler.scale(s_loss).backward(retain_graph=True)
                    s_scaler.step(s_optimizer)  # optimizer.step
                    s_scaler.update()
                    s_optimizer.zero_grad()
                    # s_scheduler.step()
                    if ema:
                        ema.update(model)
                    '''
                    4. fowarl new
                    '''
                    with amp.autocast(enabled=cuda):
                        with torch.no_grad():
                            s_pred_l = model(label_imgs)
                      
                        if hyp['loss_ota'] == 1:
                            s_loss_l_new, loss_items = compute_loss_ota(s_pred_l, label_targets.to(device), label_imgs, grad=False)  
                        else:
                            s_loss_l_new, loss_items = compute_loss(s_pred_l, label_targets.to(device), grad=False )

                        dot_product = s_loss_l_new - s_loss_l_old.detach()
                        s_loss_mpl = dot_product * s_loss_u
                        s_loss = s_loss + s_loss_mpl


                    '''
                    5. loss backward
                    loss_semi= student.forward_loss(unlabel_imgs_strong, pseudo_unlabel_targets)
                    '''

                    s_scaler.scale(s_loss).backward()
               
                    s_scaler.step(s_optimizer)  # optimizer.step
                    s_scaler.update()
                    s_optimizer.zero_grad()
                    s_scheduler.step()
                    if ema:
                        ema.update(model)
               
                    

                    model.zero_grad()

                
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    import pdb; pdb.set_trace()

       
         
            
            time_dict["optimizer"] = time.time() - start
            start = time.time()

            # Print
            if rank in [-1, 0] :
              
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
         
                s = ('%10s' * 2 + '%13.4g' * 7) % (
                    '%g/%g' % (epoch, epochs - 1), mem, s_loss_l_old, s_loss_u, s_loss, s_loss_l_new ,s_loss_mpl,  label_targets.shape[0], semi_label_items)
                pbar.set_description(s)
 
                if wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})


            if i % 50 == 0:
                json.dump(json.load(open(time_file, "r")) + [time_dict] if os.path.exists(time_file) else [time_dict], open(time_file, 'w'))
            if rank in [-1, 0] and i % round(nb/10) == 0:
           
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            
                final_epoch = epoch + 1 == epochs
                print("Evaluate EMA")
                results_1, maps, _ = test.test(data_dict,
                                                batch_size=batch_size * 2,
                                                imgsz=imgsz_test,
                                                model=deepcopy(ema.ema),
                                                single_cls=opt.single_cls,
                                                dataloader=valloader,
                                                save_dir=save_dir,
                                                verbose=nc < 50 and final_epoch,
                                                plots=False,
                                                wandb_logger=wandb_logger,
                                                compute_loss=compute_loss,
                                                is_coco=is_coco,
                                                v5_metric=opt.v5_metric, epoch=epoch)
                if epoch < hyp["supervised_epoch"]:
                    results_2 = (0,0,0,0,0,0,0)
                else:
                    print("Evaluate  Teacher")
                    results_2, maps, _ = test.test(data_dict,
                                                    batch_size=batch_size * 2,
                                                    imgsz=imgsz_test,
                                                    model=deepcopy(model_teacher.module) if is_parallel(model_teacher) else model_teacher,
                                                    single_cls=opt.single_cls,
                                                    dataloader=valloader,
                                                    save_dir=save_dir,
                                                    verbose=nc < 50 and final_epoch,
                                                    plots=False,
                                                    wandb_logger=wandb_logger,
                                                    compute_loss=compute_loss,
                                                    is_coco=is_coco,
                                                    v5_metric=opt.v5_metric, epoch=epoch)
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results_1  + '%10.4g' * 7 % results_2 + '\n')  # append metrics, val_loss
               
               
              
                tags = [
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    ]  # params
                for x, tag in zip( list(results), tags):
                    if wandb_logger.wandb:
                        wandb_logger.log({tag: x})  # W&B
                wandb_logger.end_iter()

               
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                
                # Save model
                ckpt = {'epoch': epoch,
                        'model_teacher': deepcopy(model_teacher.module if is_parallel(model_teacher) else model_teacher).half(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'optimizer': s_optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}


                # Save last, best and delete
                torch.save(ckpt, last)
                if fi > best_fitness:
                    best_fitness = fi
                    torch.save(ckpt, best)    
                if ((epoch+1) % 2) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt
                time_dict["evaluate"] = time.time() - start
                start = time.time()
            
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        time_dict["all"] = time.time() - start_all
        # torch.cuda.empty_cache()

        # DDP process 0 or single-GPU
        if rank in [-1, 0] or epoch % hyp["test_on_epoch"] == 0:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                
                results, maps, _ = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=deepcopy(ema.ema),#ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=False,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric,  epoch=epoch)

          
            with open(results_file, 'a') as f:
                f.write('%10s' * 1 % ('Epoch') \
                            + '%10s' * 7 % ('Test/Precision', 'Test/Recall', 'Test/mAP50', 'Test/mAP50:95', 'loss_bbox', 'loss_obj', 'loss_cls') + '\n')
                f.write('\n') 
                f.write("%10s" * 1 % epoch + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

        
            tags = [
                    'metrics_test/precision', 'metrics_test/recall', 'metrics_test/mAP_0.5', 'metrics_test/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
              ]  # params
            for x, tag in zip(list(results), tags):
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B


            
    
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        # if opt.global_rank in [-1, 0]:
        #     prefix = colorstr('tensorboard: ')
        #     logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        #     tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')