import os
import socket

from ptflops import get_model_complexity_info

from clip.rs import add_rs_blocks_to_clip
from utils.gr import build_adjacency_matrix
from processor.processor import do_test
from utils.logger import setup_logger
from datasets import make_dataloader
from model import build_model
from solver import make_optimizer, make_scheduler
from processor import do_train
from utils.model_utils import is_main_process, set_seed, load_clip_to_cpu, thread_flag
import torch
import torch.distributed as dist
import argparse 
from config import cfg
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils.process_utils import print_all_trainable_parameters

warnings.filterwarnings("ignore")
import time
import wandb

def main(cfg):
    # 0. Set seed
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"LOCAL_RANK: {local_rank}")
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()
    else:
        local_rank = 0 # single gpu or debug
        device = torch.device('cuda:0')
        print(f"using single GPU for training")

    # 1. Logging
    if cfg.WANDB:
        run = wandb.init(project=cfg.WANDB_PROJ, config=cfg, 
                            entity='sfrnet_', notes=socket.gethostname(), name='test', dir='./runs', job_type="training", reinit=True)
        run.name = f'{cfg.DATASETS.NAMES}-{cfg.SOLVER.OPTIMIZER_NAME}-lr{cfg.SOLVER.BASE_LR}'

    output_dir = os.path.join(cfg.OUTPUT_DIR, 'SFRNet-' + time.strftime('%Y-%m-%d-%H-%M-%S'))
    if thread_flag(cfg.MODEL.DIST_TRAIN):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger = setup_logger("SFR-Net", output_dir, if_train=True)
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info("Running with config:\n{}".format(cfg))

    # 2. Data
    train_loader, val_loader, val_loader_gzsl, train_sampler, dataset = make_dataloader(cfg)

    # 3. Model
    clip_model = load_clip_to_cpu(cfg)
    rs_model = add_rs_blocks_to_clip(clip_model, cfg) 
    model = build_model(cfg, rs_model, dataset, model_name=cfg.MODEL.NAME)

    if cfg.MODEL.LOAD:
        model.to(device)
        state_dict = torch.load(cfg.TEST.WEIGHT, map_location='cpu') 

        # remove 'module.' in state_dict's keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        if is_main_process():
            print(f"Successfully loaded weights from: {cfg.TEST.WEIGHT} into the raw model on all ranks.")

    # 4. Optimizer
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer, len(train_loader))

    ## build class matrix
    with open(os.path.join('datasets/text/', cfg.INPUT.EXP_PHASE), 'r') as f:
        class_explanation = [line.strip() for line in f.readlines()]
    class_matrix = build_adjacency_matrix(class_explanation, clip_model, threshold=cfg.INPUT.THRESHOLD)

    if is_main_process():
        print_all_trainable_parameters(model, model_name="SFR-Net")
    
    # 5. Start testing or training 
    if cfg.TEST.EVAL and cfg.MODEL.LOAD:
        if is_main_process():
            print("evaluation only mode. Skipping training.")
        do_test(cfg, model, val_loader, val_loader_gzsl, class_matrix, local_rank)
    else:
        do_train(cfg, model, train_loader, val_loader, val_loader_gzsl, class_matrix,
            optimizer, scheduler, output_dir, local_rank, train_sampler
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SFR-Net Training")
    parser.add_argument(
        "--config_file",
        help="path to config file", type=str, default="configs/sewerml_train.yml"
    )
    parser.add_argument(
        "--local_rank",
        type=int, default=0
    )
    parser.add_argument(
        "opts", 
        help="Modify config options from command-line", nargs=argparse.REMAINDER, default=None
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    main(cfg)



