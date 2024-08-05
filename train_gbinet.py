import argparse
import os
import os.path as osp
import logging
import time
import subprocess
import sys
import shutil
from tqdm import tqdm

import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import find_dataset_def
from models import find_model_def, find_loss_def
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer

from utils.logger import setup_logger
from utils.config import load_config
from utils.functions import *
import utils.depth_update as depth_update


def train_model_stage(model,
                loss_fn,
                data_loader,
                optimizer,
                scheduler,
                max_tree_depth,
                depth2stage,
                curr_epoch,
                my_rank
                ):
    avg_train_scalars = {"depth{}".format(i): DictAverageMeter() for i in range(1, max_tree_depth + 1)}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()

    with tqdm(data_loader, desc="GBi-Net Training", unit="batch") as loader:
        for iteration, sample in enumerate(loader):
            data_time = time.time() - end
            end = time.time()
            global_step = total_iteration * curr_epoch + iteration
            sample_cuda = tocuda(sample)
            
            bin_mask_prefix = torch.zeros_like(sample_cuda["masks"][str(0)]) == 0
            is_first = True
            for curr_tree_depth in range(1, max_tree_depth + 1):
                stage_id = depth2stage[str(curr_tree_depth)]
                optimizer.zero_grad()
                
                outputs = model(data_batch=sample_cuda, mode="one", stage_id=stage_id)
                    
                depth_gt = sample_cuda["depths"][str(stage_id)]
                mask = sample_cuda["masks"][str(stage_id)]
                
                depth_min_max = sample_cuda["depth_min_max"]
                gt_label, bin_mask = depth_update.get_four_label_l4_s4_bin(depth_gt, b_tree=sample_cuda["binary_tree"]["tree"], 
                    depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
                # gt_label = torch.squeeze(gt_label, 1)
                with torch.no_grad():
                    if (bin_mask_prefix.shape[1] != bin_mask.shape[1]) or (bin_mask_prefix.shape[2] != bin_mask.shape[2]):
                        bin_mask_prefix = torch.squeeze(F.interpolate(torch.unsqueeze(bin_mask_prefix.float(), 1), [bin_mask.shape[1], bin_mask.shape[2]], mode="nearest"), 1).bool()
                    bin_mask_prefix = torch.logical_and(bin_mask_prefix, bin_mask)
                    mask = torch.logical_and(bin_mask_prefix, mask > 0.0).float()

                preds = outputs["pred_feature"]
                loss = loss_fn(preds, gt_label, mask)
                loss.backward()
                optimizer.step()

                pred_label = outputs["pred_label"]

                depth_min_max = sample_cuda["depth_min_max"]
                sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                    depth_update.update_4pred_4sample1(sample_cuda["binary_tree"]["tree"], 
                    torch.unsqueeze(pred_label, 1), depth_start=depth_min_max[:, 0], depth_end=depth_min_max[:, 1], is_first=is_first)
                
                depth_est = (sample_cuda["binary_tree"]["depth"][:, 1] + sample_cuda["binary_tree"]["depth"][:, 2]) / 2.0

                is_first = False
                next_depth_stage = depth2stage[str(curr_tree_depth + 1)]
                if next_depth_stage != stage_id:
                    depth_min_max = sample_cuda["depth_min_max"]
                    sample_cuda["binary_tree"]["depth"], sample_cuda["binary_tree"]["tree"] = \
                        depth_update.depthmap2tree(depth_est, curr_tree_depth + 1, depth_start=depth_min_max[:, 0], 
                            depth_end=depth_min_max[:, 1], scale_factor=2.0, mode='bilinear')

                prob_map = outputs["pred_prob"]

                scalar_outputs = {"loss": loss}

                depth_est_mapped = mapping_color(depth_est, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")
                depth_gt_mapped = mapping_color(depth_gt, depth_min_max[:, 0], depth_min_max[:, 1], cmap="rainbow")

                image_outputs = {"depth_est": depth_est_mapped, "depth_gt": depth_gt_mapped,
                                "ref_img": sample["ref_imgs"][str(stage_id)].permute(0, 3, 1, 2) / 255.,
                                "mask": mask.cpu()}
                image_outputs["ori_mask"] = sample["masks"][str(stage_id)]
                                
                image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
                scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.0)

                scalar_outputs["accu"] = GBiNet_accu(pred_label, gt_label, mask > 0.0)
                scalar_outputs["accu0"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 0), mask > 0.0))
                scalar_outputs["accu1"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 1), mask > 0.0))
                scalar_outputs["accu2"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 2), mask > 0.0))
                scalar_outputs["accu3"] = GBiNet_accu(pred_label, gt_label, torch.logical_and(torch.eq(gt_label, 3), mask > 0.0))
                
                scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 2)
                scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 4)
                scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.0, 8)
                
                scalar_outputs["thres2mm_accu"] = 1.0 - scalar_outputs["thres2mm_error"]
                scalar_outputs["thres4mm_accu"] = 1.0 - scalar_outputs["thres4mm_error"]
                scalar_outputs["thres8mm_accu"] = 1.0 - scalar_outputs["thres8mm_error"]

                scalar_outputs["front_prob"] = Prob_mean(prob_map, mask > 0.0)
                scalar_outputs["back_prob"] = Prob_mean(prob_map, mask == 0.0)

                scalar_outputs = reduce_scalar_outputs(scalar_outputs)

                loss = tensor2float(loss)
                scalar_outputs = tensor2float(scalar_outputs)

                forward_time = time.time() - end
                avg_train_scalars["depth{}".format(curr_tree_depth)].update(scalar_outputs)

                #if iteration % log_period == 0:
                #    mode = 'train'
                #    if tensorboard_logger is not None and my_rank == 0:
                #        for key, value in scalar_outputs.items():
                #            name = '{}/{}_depth{}'.format(mode, key, curr_tree_depth)
                #            tensorboard_logger.add_scalar(name, value, global_step)
                #    
                #    logger.info(
                #    " ".join(
                #            [
                #                "Epoch {}".format(curr_epoch),
                #                "Iter {}/{}".format(iteration, total_iteration),
                #                "Max_depth {}".format(max_tree_depth),
                #                "curr_depth {}".format(curr_tree_depth),
                #                "train loss {:.4f}".format(loss),

                #                "accu {:.4f}".format(scalar_outputs["accu"]),
                #                "accu0 {:.4f}".format(scalar_outputs["accu0"]),
                #                "accu1 {:.4f}".format(scalar_outputs["accu1"]),
                #                "accu2 {:.4f}".format(scalar_outputs["accu2"]),
                #                "accu3 {:.4f}".format(scalar_outputs["accu3"]),

                #                "abs_depth_error {:.4f}".format(scalar_outputs["abs_depth_error"]),
                #                "thres2mm_error {:.4f}".format(scalar_outputs["thres2mm_error"]),
                #                "thres4mm_error {:.4f}".format(scalar_outputs["thres4mm_error"]),
                #                "thres8mm_error {:.4f}".format(scalar_outputs["thres8mm_error"]),
                #                "thres2mm_accu {:.4f}".format(scalar_outputs["thres2mm_accu"]),
                #                "thres4mm_accu {:.4f}".format(scalar_outputs["thres4mm_accu"]),
                #                "thres8mm_accu {:.4f}".format(scalar_outputs["thres8mm_accu"]),
                #                "front_prob {:.4f}".format(scalar_outputs["front_prob"]),
                #                "back_prob {:.4f}".format(scalar_outputs["back_prob"]),
                #                "forward_time {:.4f}".format(forward_time),
                #                "data_time {:.4f}".format(data_time),
                #                "lr {}".format([a["lr"] for a in optimizer.param_groups]),
                #                "max_mem: {:.0f}".format(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                #            ]
                #        )
                #    )

                end = time.time()

                loader.set_postfix(
                        loss=f"{(loss/(iteration+1)):6.6f}",
                        max_memory=f"{(torch.cuda.max_memory_allocated() / (1024.0 ** 2)):.0f}"
                        )

        if scheduler is not None:
            scheduler.step_update(curr_epoch * total_iteration + iteration)
        end = time.time()

def train(rank, cfg):
    torch.cuda.set_device(rank)
        
    set_random_seed(cfg["random_seed"])
    state_dict = None
    
    model_def = find_model_def(cfg["model_file"], cfg["model_name"])
    model = model_def(cfg).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss_def = find_loss_def(cfg["model_file"], cfg.get("loss_name", cfg["model_name"] + "_loss"))
    model_loss = loss_def
    
    optimizer = build_optimizer(cfg, model)

    if cfg.get("img_mean") and cfg.get("img_std"):
        img_mean = cfg.get("img_mean")
        img_std = cfg.get("img_std")
    else:
        img_mean = None
        img_std = None

    MVSDataset = find_dataset_def(cfg["dataset"])
    
    train_dataset = MVSDataset(cfg["data"]["train"]["root_dir"], 
        cfg["data"]["train"]["listfile"], "train", 
        cfg["data"]["train"]["num_view"], 
        cfg["data"]["train"]["num_depth"], 
        cfg["data"]["train"]["interval_scale"],
        img_mean=None, img_std=None,
        out_scale=cfg["data"]["out_scale"],
        self_norm=cfg["data"]["train"]["self_norm"],
        color_mode=cfg["data"]["train"]["color_mode"],
        is_stage=cfg["model"].get("is_stage", False), 
        stage_info=cfg["model"].get("stage_info", None),
        random_view=cfg["data"]["train"].get("random_view", False),
        img_interp=cfg["data"]["train"].get("img_interp", "linear"),
        random_crop=cfg["data"]["train"].get("random_crop", False),
        crop_h=cfg["data"]["train"].get("crop_h", 512),
        crop_w=cfg["data"]["train"].get("crop_w", 640),
        depth_num=cfg["data"]["train"].get("depth_num", 4),
        transform=cfg["data"]["train"].get("transform", True))

    world_size=cfg["world_size"]
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_data_loader = DataLoader(train_dataset, cfg["train"]["batch_size"], sampler=train_sampler, num_workers=cfg["data"]["num_workers"])
    
    scheduler = build_scheduler(cfg, optimizer, len(train_data_loader))

    start_epoch = 0
    
    if cfg["scheduler"].get("change"):
        # now only for multi-step scheduler
        from collections import Counter
        logger.info("Changing scheduler ...")
        scheduler.milestones=Counter(cfg["scheduler"]["milestones"])
        scheduler.gamma=cfg["scheduler"]["gamma"]
    
    # train    
    max_epoch = cfg["train"]["max_epoch"]
    ckpt_period = cfg["train"]["checkpoint_period"]

    max_max_tree_depth = cfg["max_depth"]
    for epoch in range(start_epoch, max_epoch):
        train_sampler.set_epoch(epoch)

        init_max_depth = cfg["model"].get("init_max_depth", 2)
        max_tree_depth = min(epoch + init_max_depth, max_max_tree_depth)
        train_func = train_model_stage
        train_func(model,
            model_loss,
            data_loader=train_data_loader,
            optimizer=optimizer,
            scheduler=None if cfg["scheduler"]["name"] == 'multi_step' else scheduler,
            max_tree_depth=max_tree_depth,
            depth2stage=cfg["model"]["stage_info"]["depth2stage"],
            curr_epoch=epoch,
            my_rank=rank
            )

        if cfg["scheduler"]["name"] == 'multi_step':
            scheduler.step()

        # checkpoint
        if epoch % ckpt_period == 0 or (epoch + 1) == max_epoch:
            if scheduler is not None:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    best_metric_name: best_metric
                    },
                    osp.join(output_dir, "model_{:03d}.ckpt".format(epoch)))
            else:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    best_metric_name: best_metric
                    },
                    osp.join(output_dir, "model_{:03d}.ckpt".format(epoch)))
        
def main():
    parser = argparse.ArgumentParser(description="PyTorch GBiNet Training")
    parser.add_argument("--cfg", dest="config_file", default="", metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config_file)

    gpu_index = 0
    cfg["world_size"] = 1
    train(gpu_index,cfg)

if __name__ == "__main__":
    main()
