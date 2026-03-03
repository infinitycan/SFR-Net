import logging
import os
import time
import wandb
import datetime
import numpy as np
import torch
import torch.distributed as dist

from torch.cuda import amp
from loss.scl_loss import SynergisticContrastiveLoss
from utils.model_utils import is_main_process, thread_flag
from utils.model_utils import ModelEma
from utils.meter import AverageMeter
from utils.metrics import compute_map, multilabel_evaluation

import warnings
warnings.filterwarnings("ignore")


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        val_loader_gzsl,
        class_matrix,
        optimizer,
        scheduler,
        output_dir,
        local_rank,
        train_sampler=None,
    ):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = torch.device("cuda", local_rank)
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("SFR-Net.train")
    logger.info('start training')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            if is_main_process():
                print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    scaler = amp.GradScaler()

    criterion = SynergisticContrastiveLoss(temperature=cfg.MODEL.TEMPERATURE, margin=cfg.MODEL.MARGIN, gamma=cfg.MODEL.GAMMA)

    # Training
    label_smoothing = cfg.SOLVER.LABEL_SMOOTHING
    torch.cuda.empty_cache()
    ema_m = None
    tot_iters = len(train_loader) * epochs

    for epoch in range(1, epochs + 1):
        if cfg.MODEL.USE_EMA:
            if cfg.MODEL.DIST_TRAIN:
                ema_m = ModelEma(model.module, cfg.MODEL.EMA_DECAY, device=dist.get_rank())
            else:
                ema_m = ModelEma(model, cfg.MODEL.EMA_DECAY, device=device)
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        loss_meter.reset()
        scheduler.step(epoch)
        
        model.train()
        cnt = 0
        for n_iter, (img, label) in enumerate(train_loader):
            start = time.time()
            if cfg.SOLVER.LR_SCHEDULER == 'onecycle':
                scheduler.step()
            optimizer.zero_grad()

            img = img.to(device)
            class_matrix = class_matrix.to(device)

            if label_smoothing:
                label_f = label.float()
                label_soft = torch.where(label_f == 1, torch.tensor(0.9), label_f)
                label_soft = torch.where(label_soft == 0, torch.tensor(0.1), label_soft)
                target = label_soft.to(device)
            else:
                target = label.to(device)

            with amp.autocast(enabled=True):
                score = model(img, class_matrix, zsl=False, gzsl=False)
                loss = criterion(score, target)
  
            scaler.scale(loss).backward()
            gpu_mem = torch.cuda.max_memory_allocated()/(1024.0 * 1024.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 添加梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            if ema_m is not None:
                if cfg.MODEL.DIST_TRAIN:
                    ema_m.update(model.module)
                else:
                    ema_m.update(model)

            label = label.numpy()
            outputs_np = score.data.cpu().numpy()

            if cnt == 0:
                g_labels = label
                p_score = outputs_np
            else:
                g_labels = np.row_stack((g_labels, label))
                p_score = np.row_stack((p_score, outputs_np))

            cnt += 1

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()

            batch_time.update(time.time() - start)

            # Logging
            if (n_iter + 1) % log_period == 0:
                if thread_flag(cfg.MODEL.DIST_TRAIN):
                    now_iter = (epoch-1) * len(train_loader) + n_iter
                    nb_remain = tot_iters - now_iter
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                    cur_lr = optimizer.param_groups[1]['lr']

                    mAP, APs = compute_map(p_score, g_labels)

                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, mAP: {:.3%}, lr: {:.2e}, mem: {:.2f}MB, speed:{:.2f}[img/s], ETA: {}"
                        .format(epoch, n_iter+1, len(train_loader), loss_meter.avg, mAP, cur_lr, gpu_mem, train_loader.batch_size/batch_time.avg, eta))

                    if cfg.WANDB:
                        wandb.log({
                            "epoch": epoch,
                            "lr": cur_lr,
                            "train loss": loss_meter.avg,
                            "train mAP": mAP
                        })
    
        if cfg.SOLVER.SAVE_MODEL:
            if thread_flag(cfg.MODEL.DIST_TRAIN):
                output_path = os.path.join(output_dir, f'epoch{epoch}.pth')
                torch.save(model.module.state_dict(), output_path) # save ddp model weights
    
        # Testing
        if epoch % eval_period == 0:
            if is_main_process():
                logger.info('================start testing==================')

            mAP_zsl, APs_zsl, Result_k, Result_k2 = validate(cfg, val_loader, class_matrix, model, device, zsl=True, gzsl=False) # ML-ZSL
            mAP, APs, Result_k_gzsl, Result_k2_gzsl = validate(cfg, val_loader_gzsl, class_matrix, model, device, zsl=False, gzsl=True) # ML-GZSL

            if ema_m is not None:
                mAP_ema_zsl, Result_k_ema, Result_k2_ema = validate(cfg, val_loader, ema_m.module, device, zsl=True, gzsl=False)
                mAP_ema, Result_k_gzsl_ema, Result_k2_gzsl_ema = validate(cfg, val_loader_gzsl, ema_m.module, device, zsl=False, gzsl=True)

            if is_main_process():   
                logger.info("Test Results - Epoch: {}".format(epoch))
                logger.info("ZSL:")
                logger.info("APs: %s", ", ".join(f"{ap:.3%}" for ap in APs_zsl))
                logger.info("mAP: {:.3%}".format(mAP_zsl))
                logger.info("OP_{}: {:.3%}, OR_{}: {:.3%}, OF1_{}: {:.3%}".format(cfg.INPUT.TOP_K_ZSL[0], Result_k['OP'], cfg.INPUT.TOP_K_ZSL[0], Result_k['OR'], cfg.INPUT.TOP_K_ZSL[0], Result_k['OF1']))
                logger.info("OP_{}: {:.3%}, OR_{}: {:.3%}, OF1_{}: {:.3%}".format(cfg.INPUT.TOP_K_ZSL[1], Result_k2['OP'], cfg.INPUT.TOP_K_ZSL[1], Result_k2['OR'], cfg.INPUT.TOP_K_ZSL[1], Result_k2['OF1']))
                logger.info("-------------------------------")
                logger.info("GZSL:")
                logger.info("Seen APs: %s", ", ".join(f"{ap:.3%}" for ap in APs[:-5]))
                logger.info("Unseen APs: %s", ", ".join(f"{ap:.3%}" for ap in APs[-5:]))
                logger.info("mAP: {:.3%}".format(mAP))
                logger.info("OP_{}: {:.3%}, OR_{}: {:.3%}, OF1_{}: {:.3%}".format(cfg.INPUT.TOP_K_GZSL[0], Result_k_gzsl['OP'], cfg.INPUT.TOP_K_GZSL[0], Result_k_gzsl['OR'], cfg.INPUT.TOP_K_GZSL[0], Result_k_gzsl['OF1']))
                logger.info("OP_{}: {:.3%}, OR_{}: {:.3%}, OF1_{}: {:.3%}".format(cfg.INPUT.TOP_K_GZSL[1], Result_k2_gzsl['OP'], cfg.INPUT.TOP_K_GZSL[1], Result_k2_gzsl['OR'], cfg.INPUT.TOP_K_GZSL[1], Result_k2_gzsl['OF1']))
                logger.info("-------------------------------")

                if ema_m is not None:
                    logger.info("EMA Results:")
                    logger.info("ZSL:")
                    logger.info("mAP: {:.3%}".format(mAP_ema_zsl))
                    logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k_ema['OP'], Result_k_ema['OR'], Result_k_ema['OF1']))
                    logger.info("OP_5: {:.3%}, OR_5: {:.3%}, OF1_5: {:.3%}".format(Result_k2_ema['OP'], Result_k2_ema['OR'], Result_k2_ema['OF1']))
                    logger.info("GZSL:")
                    logger.info("mAP: {:.3%}".format(mAP_ema))
                    logger.info("OP_5: {:.3%}, OR_5: {:.3%}, OF1_5: {:.3%}".format(Result_k_gzsl_ema['OP'], Result_k_gzsl_ema['OR'], Result_k_gzsl_ema['OF1']))
                    logger.info("OP_10: {:.3%}, OR_10: {:.3%}, OF1_10: {:.3%}".format(Result_k2_gzsl_ema['OP'], Result_k2_gzsl_ema['OR'], Result_k2_gzsl_ema['OF1']))

                # 3. log wandb
                if cfg.WANDB:
                    wandb.log({
                        "mAP_zsl": mAP_zsl,
                        "mAP": mAP,
                        "OP_1": Result_k['OP'], "OR_1": Result_k['OR'], "OF1_1": Result_k['OF1'],
                        "OP_3": Result_k2['OP'], "OR_3": Result_k2['OR'], "OF1_3": Result_k2['OF1'],
                        "OP_5 GZSL": Result_k_gzsl['OP'], "OR_5 GZSL": Result_k_gzsl['OR'], "OF1_5 GZSL": Result_k_gzsl['OF1'],
                        "OP_10 GZSL": Result_k2_gzsl['OP'], "OR_10 GZSL": Result_k2_gzsl['OR'], "OF1_10 GZSL": Result_k2_gzsl['OF1'],
                    })
                    if ema_m is not None:
                        wandb.log({
                            "OP_3 ema": Result_k_ema['OP'], "OR_3 ema": Result_k_ema['OR'], "OF1_3 ema": Result_k_ema['OF1'],
                            "OP_5 ema": Result_k2_ema['OP'], "OR_5 ema": Result_k2_ema['OR'], "OF1_5 ema": Result_k2_ema['OF1'],
                            "OP_3 GZSL ema": Result_k_gzsl_ema['OP'], "OR_3 GZSL ema": Result_k_gzsl_ema['OR'], "OF1_3 GZSL ema": Result_k_gzsl_ema['OF1'],
                            "OP_5 GZSL ema": Result_k2_gzsl_ema['OP'], "OR_5 GZSL ema": Result_k2_gzsl_ema['OR'], "OF1_5 GZSL ema": Result_k2_gzsl_ema['OF1'],
                        })

                logger.info('================testing ended==================')

            torch.cuda.empty_cache()

@torch.no_grad()
def validate(cfg, val_loader, class_matrix, model, device, zsl=True, gzsl=False):
    model.eval()
    gpu_local_preds = [] 
    gpu_local_labels = [] 
    batch_time = AverageMeter()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    for n_iter, (img, label) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            target = label.to(device) # target 已在 device 上
            start = time.time()

            score = model(img, class_matrix, zsl=zsl, gzsl=gzsl)

            batch_time.update(time.time() - start)
            gpu_local_preds.append(score.detach()) 
            gpu_local_labels.append(target.detach()) 

        if (n_iter + 1) % cfg.SOLVER.LOG_PERIOD == 0 and rank == 0:
            mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"mem: {mem:.2f}MB, test speed: {val_loader.batch_size / batch_time.avg:.2f} img/s")

    final_local_preds_gpu = torch.cat(gpu_local_preds, dim=0)
    final_local_labels_gpu = torch.cat(gpu_local_labels, dim=0)

    if world_size > 1:
        if rank == 0:
            gathered_preds_list_gpu = [torch.empty_like(final_local_preds_gpu) for _ in range(world_size)]
            gathered_labels_list_gpu = [torch.empty_like(final_local_labels_gpu) for _ in range(world_size)]
        else:
            gathered_preds_list_gpu = None
            gathered_labels_list_gpu = None
        
        dist.gather(final_local_preds_gpu, gather_list=gathered_preds_list_gpu, dst=0)
        dist.gather(final_local_labels_gpu, gather_list=gathered_labels_list_gpu, dst=0)
        
        dist.barrier()
            
    if rank == 0:
        if world_size > 1:
            all_preds_concatenated = torch.cat(gathered_preds_list_gpu, dim=0).cpu().numpy()
            all_labels_concatenated = torch.cat(gathered_labels_list_gpu, dim=0).cpu().numpy()
        else:
            all_preds_concatenated = final_local_preds_gpu.cpu().numpy()
            all_labels_concatenated = final_local_labels_gpu.cpu().numpy()

        mAP, APs = compute_map(all_preds_concatenated, all_labels_concatenated)

        if zsl:
            Result_k = multilabel_evaluation(all_preds_concatenated, all_labels_concatenated, k=cfg.INPUT.TOP_K_ZSL[0])
            Result_k2 = multilabel_evaluation(all_preds_concatenated, all_labels_concatenated, k=cfg.INPUT.TOP_K_ZSL[1])
        else:
            Result_k = multilabel_evaluation(all_preds_concatenated, all_labels_concatenated, k=cfg.INPUT.TOP_K_GZSL[0])
            Result_k2 = multilabel_evaluation(all_preds_concatenated, all_labels_concatenated, k=cfg.INPUT.TOP_K_GZSL[1])

        return mAP, APs, Result_k, Result_k2
    else:
        return None, None, None, None

def do_test(
        cfg,
        model,
        val_loader,
        val_loader_gzsl,
        class_matrix,
        local_rank
    ):
    device = torch.device("cuda", local_rank)
    logger = logging.getLogger("SFR-Net.test")
    logger.info('start testing')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            if is_main_process():
                print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
            print(f"Rank {dist.get_rank()} running on device {torch.cuda.current_device()}")


    # model.eval()
    class_matrix = class_matrix.to(device)
    if is_main_process():
        logger.info('================start testing==================')

    mAP_zsl, APs_zsl, Result_k, Result_k2 = validate(cfg, val_loader, class_matrix, model, device, zsl=True, gzsl=False) 
    mAP, APs, Result_k_gzsl, Result_k2_gzsl = validate(cfg, val_loader_gzsl, class_matrix, model, device, zsl=False, gzsl=True)

    if is_main_process():   
        logger.info("Validation Results")
        logger.info("ZSL:")
        logger.info("APs: %s", ", ".join(f"{ap:.3%}" for ap in APs_zsl))
        logger.info("mAP: {:.3%}".format(mAP_zsl))
        logger.info("OP_1: {:.3%}, OR_1: {:.3%}, OF1_1: {:.3%}".format(Result_k['OP'], Result_k['OR'], Result_k['OF1']))
        logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k2['OP'], Result_k2['OR'], Result_k2['OF1']))
        logger.info("GZSL:")
        logger.info("Seen APs: %s", ", ".join(f"{ap:.3%}" for ap in APs[:-5]))
        logger.info("Unseen APs: %s", ", ".join(f"{ap:.3%}" for ap in APs[-5:]))
        logger.info("mAP: {:.3%}".format(mAP))
        logger.info("OP_1: {:.3%}, OR_1: {:.3%}, OF1_1: {:.3%}".format(Result_k_gzsl['OP'], Result_k_gzsl['OR'], Result_k_gzsl['OF1']))
        logger.info("OP_3: {:.3%}, OR_3: {:.3%}, OF1_3: {:.3%}".format(Result_k2_gzsl['OP'], Result_k2_gzsl['OR'], Result_k2_gzsl['OF1']))

        logger.info('================testing ended==================')

        torch.cuda.empty_cache()





