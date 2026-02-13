#!/usr/bin/env python
import os
import pdb
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from base.baseTrainer import load_state_dict

import math
import cv2
from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models import get_model
from torch.optim.lr_scheduler import StepLR, LambdaLR

# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)

import warnings
warnings.filterwarnings("ignore")
import wandb


def _unwrap_model(model):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        return model.module
    return model


def _snapshot_trainable_params(model, exclude_prefixes=None):
    module = _unwrap_model(model)
    exclude_prefixes = exclude_prefixes or []
    init_params = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        init_params[name] = p.detach().clone()
    return init_params


def _l2sp_loss(model, init_params):
    if not init_params:
        return torch.zeros((), device=next(_unwrap_model(model).parameters()).device)
    module = _unwrap_model(model)
    loss = torch.zeros((), device=next(module.parameters()).device)
    for name, p in module.named_parameters():
        if name in init_params:
            loss = loss + (p - init_params[name]).pow(2).mean()
    return loss



def main():
    args = get_parser()

    #os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    #initialize wandb 
    wandb.init(project="multitalk_custom_s2", name=args.save_path.split("/")[-1],dir="logs")
    wandb.config.update(args)

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    cfg.gpu = gpu

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
        

    # ####################### Data Loader ####################### #
    from dataset.data_loader_joint_data_batched import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']

    if cfg.evaluate:
        val_loader = dataset['valid']
        test_loader = dataset['test']

    val_loss_log = 1000
    
    # ####################### Model ####################### #
    global logger
    logger = get_logger()

    model = get_model(cfg)
    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")

    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()

    trainable_modules = getattr(cfg, "trainable_modules", [])
    if getattr(cfg, "finetune_expr_affine", False):
        module = _unwrap_model(model)
        for name, p in module.named_parameters():
            p.requires_grad = name in {"expr_scale", "expr_bias"}
    elif getattr(cfg, "finetune_expr_adapter", False):
        module = _unwrap_model(model)
        for name, p in module.named_parameters():
            p.requires_grad = name.startswith("expr_adapter") or name == "expr_adapter_scale"
    elif trainable_modules:
        module = _unwrap_model(model)
        for name, p in module.named_parameters():
            p.requires_grad = any(name.startswith(prefix) for prefix in trainable_modules)

    if getattr(cfg, "freeze_audio_encoder", False) and hasattr(_unwrap_model(model), "audio_encoder"):
        for p in _unwrap_model(model).audio_encoder.parameters():
            p.requires_grad = False

    # ####################### Loss ############################# #
    loss_fn = nn.MSELoss()
    
    # ####################### Optimizer ######################## #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # =========== Load checkpoint ===========
    checkpoint_path = getattr(cfg, "weight", None) or getattr(cfg, "resume", None)
    if not checkpoint_path:
        raise ValueError("Missing base checkpoint path. Set TRAIN.weight (or TRAIN.resume) in the yaml.")
    print("=> Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cpu())
    load_state_dict(model, checkpoint['state_dict'], strict=False)
    print("=> Loaded checkpoint '{}'".format(checkpoint_path))
    # =========== End load checkpoint ===========

    l2sp_weight = getattr(cfg, "l2sp_weight", 0.0)
    l2sp_exclude = getattr(cfg, "l2sp_exclude", [])
    init_params = _snapshot_trainable_params(model, exclude_prefixes=l2sp_exclude) if l2sp_weight > 0 else {}

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        loss_train, blendshapes_loss_train, codebook_loss_train, nt_xent_loss_train, reg_loss_train, l2sp_loss_train = train(train_loader, model, loss_fn, optimizer, epoch, cfg, init_params, l2sp_weight)
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
            print("Updated learning rate to {}".format(scheduler.get_last_lr()[0]))
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'codebook_loss_train: {} '
                        'nt_xent_loss_train: {} '
                        'blendshapes_loss_train: {} '
                        .format(epoch_log, loss_train, codebook_loss_train, nt_xent_loss_train, blendshapes_loss_train)
                        )

        wandb.log({"loss_train": loss_train, "blendshapes_loss_train": blendshapes_loss_train, "codebook_loss_train": codebook_loss_train, "nt_xent_loss_train": nt_xent_loss_train, "reg_loss_train": reg_loss_train, "l2sp_loss_train": l2sp_loss_train}, epoch_log)

        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val = validate(val_loader, model, loss_fn, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {} '
                            .format(epoch_log, loss_val)
                            )
            wandb.log({"loss_val": loss_val}, epoch_log)
            
            save_checkpoint(model,
                            sav_path=os.path.join(cfg.save_path, 'model_'+str(epoch_log)),
                            stage=2
                            )

def train(train_loader, model, loss_fn, optimizer, epoch, cfg, init_params=None, l2sp_weight=0.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_blendshapes_meter = AverageMeter()
    loss_codebook_meter = AverageMeter()
    nt_xent_loss = AverageMeter()
    loss_reg_meter = AverageMeter()
    l2sp_meter = AverageMeter()
    

    model.train()
    model.autoencoder.eval()
    if hasattr(model, "audio_encoder") and getattr(cfg, "wav2vec_unfreeze_layers", 0) == 0:
        model.audio_encoder.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"----> Total trainable parameters: {trainable_params}")

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)

    for i, (padded_blendshapes, blendshape_mask, padded_audios, audio_mask) in enumerate(train_loader):
        ####################
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)

        #################### cpu to gpu
        padded_blendshapes  = padded_blendshapes.cuda(cfg.gpu, non_blocking=True)
        blendshape_mask     = blendshape_mask.cuda(cfg.gpu, non_blocking=True)
        padded_audios       = padded_audios.cuda(cfg.gpu, non_blocking=True)
        audio_mask          = audio_mask.cuda(cfg.gpu, non_blocking=True)

        loss, loss_detail = model(
            padded_blendshapes,
            blendshape_mask,
            padded_audios,
            audio_mask,
            criterion=loss_fn,
        )

        l2sp_loss = _l2sp_loss(model, init_params) if l2sp_weight > 0 else torch.zeros((), device=loss.device)
        total_loss = loss + l2sp_weight * l2sp_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ######################
        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_meter, loss_blendshapes_meter, loss_codebook_meter, nt_xent_loss, loss_reg_meter],
            [loss, loss_detail[0],  loss_detail[1], loss_detail[2], loss_detail[3]]):
            m.update(x.item(), 1)
        l2sp_meter.update(l2sp_loss.item(), 1)

        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        'codebook_loss_meter: {loss_codebook_meter.val:.4f} '
                        'nt_xent_loss: {nt_xent_loss.val:.4f} '
                        'loss_blendshapes_meter: {loss_blendshapes_meter.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter,
                            loss_codebook_meter=loss_codebook_meter,
                                nt_xent_loss=nt_xent_loss,
                                loss_blendshapes_meter=loss_blendshapes_meter
                                ))

    return loss_meter.avg, loss_blendshapes_meter.avg, loss_codebook_meter.avg, nt_xent_loss.avg, loss_reg_meter.avg, l2sp_meter.avg

def validate(val_loader, model, loss_fn, cfg):
    loss_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (padded_blendshapes, blendshape_mask, padded_audios, audio_mask) in enumerate(val_loader):
            padded_blendshapes  = padded_blendshapes.cuda(cfg.gpu, non_blocking=True)
            blendshape_mask     = blendshape_mask.cuda(cfg.gpu, non_blocking=True)
            padded_audios       = padded_audios.cuda(cfg.gpu, non_blocking=True)
            audio_mask          = audio_mask.cuda(cfg.gpu, non_blocking=True)

            loss, loss_detail = model(padded_blendshapes,blendshape_mask,padded_audios,audio_mask,criterion=loss_fn)
            loss_meter.update(loss.item(), 1)

    return loss_meter.avg

if __name__ == '__main__':
    main()
