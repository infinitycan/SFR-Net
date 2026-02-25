import os
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing
import cv2
import numpy as np
from PIL import Image

from datasets.multilabel.sewerml_classification import SewerMLZSMultiLabelClassification
from utils.model_utils import is_main_process, thread_flag

from .bases import ImageDataset
from datasets.multilabel.wzpipe_classification import WZPipeZSMultiLabelClassification

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.set_num_threads(1)

__factory = {
    'SewerML': SewerMLZSMultiLabelClassification,
    'WZPipe': WZPipeZSMultiLabelClassification,
}

def train_collate_fn(batch):
    imgs, label = zip(*batch)
    label = torch.tensor(label, dtype=torch.float32) 
    return torch.stack(imgs, dim=0), label


def val_collate_fn(batch):
    imgs, label = zip(*batch)
    label = torch.tensor(label, dtype=torch.float32) 
    return torch.stack(imgs, dim=0), label

def make_dataloader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, dataset_type = cfg.DATASETS.TYPE)
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    train_set = ImageDataset(dataset.train, transform=train_transforms, mirror=True, Aug=True)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    val_set = ImageDataset(dataset.test, transform=val_transforms, mirror=False, Aug=False)
    val_set_gzsl = ImageDataset(dataset.test_gzsl, transform=val_transforms, mirror=False, Aug=False)

    if cfg.MODEL.DIST_TRAIN:
        if is_main_process():
            print('DIST_TRAIN START')
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
        mini_batch_size_test = cfg.TEST.IMS_PER_BATCH // dist.get_world_size()
        if is_main_process():
            print('===========================\n mini batch size:', mini_batch_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        valid_sample = torch.utils.data.distributed.DistributedSampler(val_set)
        valid_gzsl_sample = torch.utils.data.distributed.DistributedSampler(val_set_gzsl)
        nw = min([mini_batch_size if mini_batch_size > 1 else 0, 8])
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=mini_batch_size,
            pin_memory=True,
            num_workers=nw,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=train_collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=mini_batch_size_test,
            shuffle=False,
            sampler=valid_sample,
            num_workers=nw,
            collate_fn=val_collate_fn,
            drop_last=True
        )
        val_loader_gzsl = DataLoader(
            val_set_gzsl, 
            batch_size=mini_batch_size_test, 
            shuffle=False,
            sampler=valid_gzsl_sample,
            num_workers=nw,
            collate_fn=val_collate_fn,
            drop_last=True
        )
        return train_loader, val_loader, val_loader_gzsl, train_sampler, dataset
    else:
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.SOLVER.IMS_PER_BATCH, 
            shuffle=False,
            num_workers=num_workers, 
            collate_fn=train_collate_fn, 
            drop_last=True, 
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            persistent_workers=True,
        )
        val_loader_gzsl = DataLoader(
            val_set_gzsl, 
            batch_size=cfg.TEST.IMS_PER_BATCH, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            persistent_workers=True,
        )

        if thread_flag(cfg.MODEL.DIST_TRAIN):
            print('Data Loading Done!')

        return train_loader, val_loader, val_loader_gzsl, None, dataset

