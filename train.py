import argparse
import importlib
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import SCDDataset, get_dataset_spec
from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, SoftSemanticConsistency
from utils.seed import set_seed
from utils.metrics import accuracy, SCDD_eval_from_hist, AverageMeter, get_hist


ENCODER_REGISTRY = {
    "pera": "models.pera",
    "vmambaB": "models.vmamba",
    "resnet50": "models.resnet",
    "swinV2L": "models.swin",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Semantic Change Detection training script.")

    # Model / Encoder
    parser.add_argument("--encoder", type=str, default="PerA", choices=list(ENCODER_REGISTRY.keys()), help="Encoder backbone name.")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to pretrained backbone weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze encoder backbone during training.")

    # Dataset
    parser.add_argument("--data-name", type=str, default="SECOND", help="Dataset name. Example: SECOND or LandsatSCD.")
    parser.add_argument("--data-path", type=str, required=True, help="Root directory of dataset.")
    parser.add_argument("--input-size", type=int, default=448, help="Input image size before decoder.")
    parser.add_argument("--output-size", type=int, default=512, help="Final output prediction size.")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of dataloader workers.")
    parser.add_argument("--norm-profile", type=str, default="auto", choices=["auto", "imagenet", "pera"], help="Normalization profile. 'auto' selects normalization based on encoder.")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Minimum learning rate after decay.")
    parser.add_argument("--lr-decay-power", type=float, default=1.5, help="Polynomial learning rate decay power.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio of total iterations.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--clip-grad", type=float, default=1.5, help="Gradient clipping max norm.")
    parser.add_argument("--drop-rate", type=float, default=0.3, help="Drop path / dropout rate.")
    parser.add_argument("--tau", type=float, default=0.01, help="Temperature parameter for semantic consistency loss.")
    parser.add_argument("--seed", type=int, default=3701, help="Random seed.")
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "none"], help="AMP mixed precision type.")

    # Resume / Logging
    parser.add_argument("--load-path", type=str, default=None, help="Checkpoint path for resuming training.")
    parser.add_argument("--log-root", type=str, default="./logs", help="Root directory for logs and checkpoints.")
    parser.add_argument("--note", type=str, default="", help="Extra note appended to log directory name.")

    # Dataloader
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor for dataloader workers.")

    return parser.parse_args()


def set_visible_gpu(gpu):
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def build_model(args):
    module = importlib.import_module(ENCODER_REGISTRY[args.encoder])
    return module.build_model(
        num_classes=args.num_classes,
        input_size=args.input_size,
        output_size=args.output_size,
        drop_rate=args.drop_rate,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone,
    )


def amp_context(args):
    if args.amp_dtype in ["none", "fp32"]:
        return torch.float32
    if args.amp_dtype == "bf16":
        return torch.bfloat16
    if args.amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {args.amp_dtype}")


def make_log_dir(args):
    note = f"_{args.note}" if args.note else ""
    log_dir = Path(args.log_root) / args.data_name / args.encoder / f"{time.strftime('%y%m%d%H%M%S')}{note}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def save_ckpt(path, epoch, net, optimizer, best_fscd, sek, miou):
    torch.save({
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "Fscd": best_fscd,
        "Sek": sek,
        "mIoU": miou,
    }, path)
    torch.cuda.empty_cache()
    print(f"Checkpoint saved to {path}")


def resume_ckpt(net, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    net.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Checkpoint loaded from {path}")
    torch.cuda.empty_cache()
    return net, optimizer, checkpoint["epoch"] + 1, checkpoint["Fscd"]


def adjust_lr(optimizer, iter_ratio, init_lr, warmup_ratio, min_lr, decay_power):
    if iter_ratio < warmup_ratio:
        lr = init_lr * (iter_ratio / warmup_ratio)
    else:
        scale = ((1.0 - iter_ratio) / (1.0 - warmup_ratio)) ** decay_power
        lr = min_lr + (init_lr - min_lr) * scale

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def build_loaders(args):
    train_set = SCDDataset(
        root=args.data_path,
        mode="train",
        dataset_name=args.data_name,
        encoder=args.encoder,
        norm_profile=args.norm_profile,
    )

    val_set = SCDDataset(
        root=args.data_path,
        mode="test",
        dataset_name=args.data_name,
        encoder=args.encoder,
        norm_profile=args.norm_profile,
    )

    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader


def train_one_epoch(args, train_loader, net, criterion, criterion_sc, optimizer, scaler, curr_epoch, all_iters):
    net.train()
    train_seg_loss = AverageMeter()
    train_bn_loss = AverageMeter()
    train_sc_loss = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(tqdm(train_loader, ncols=80)):
        running_iter = curr_iter + i + 1
        iter_ratio = running_iter / all_iters
        adjust_lr(
            optimizer,
            iter_ratio,
            init_lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            min_lr=args.min_lr,
            decay_power=args.lr_decay_power,
        )

        imgs_a, imgs_b, labels_a, labels_b, _ = data
        imgs_a = imgs_a.cuda(non_blocking=True).float()
        imgs_b = imgs_b.cuda(non_blocking=True).float()
        labels_bn = (labels_a > 0).unsqueeze(1).cuda(non_blocking=True).float()
        labels_a = labels_a.cuda(non_blocking=True).long()
        labels_b = labels_b.cuda(non_blocking=True).long()

        with autocast(dtype=amp_context(args)):
            out_change, outputs_a, outputs_b = net(imgs_a, imgs_b)
            loss_seg = criterion(outputs_a, labels_a) + criterion(outputs_b, labels_b)
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_a[:, 1:], outputs_b[:, 1:], labels_bn)
            loss = (loss_seg * 0.5 + loss_bn + loss_sc)
            if args.grad_accum_steps > 1:
                loss /= args.grad_accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % args.grad_accum_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_seg_loss.update(loss_seg.detach().float().cpu().numpy())
        train_bn_loss.update(loss_bn.detach().float().cpu().numpy())
        train_sc_loss.update(loss_sc.detach().float().cpu().numpy())

    return train_seg_loss.avg, train_bn_loss.avg, train_sc_loss.avg


@torch.no_grad()
def validate(args, val_loader, net, criterion):
    net.eval()
    torch.cuda.empty_cache()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()
    hist = np.zeros((args.num_classes, args.num_classes))
    start = time.time()

    for data in tqdm(val_loader, ncols=80):
        imgs_a, imgs_b, labels_a, labels_b, _ = data
        imgs_a = imgs_a.cuda().float()
        imgs_b = imgs_b.cuda().float()
        labels_a = labels_a.cuda().long()
        labels_b = labels_b.cuda().long()

        with torch.no_grad():
            out_change, outputs_a, outputs_b = net(imgs_a, imgs_b)
            loss = 0.5 * criterion(outputs_a, labels_a) + 0.5 * criterion(outputs_b, labels_b)

        val_loss.update(loss.cpu().detach().numpy())

        labels_a_np = labels_a.cpu().detach().numpy()
        labels_b_np = labels_b.cpu().detach().numpy()
        outputs_a_cpu = outputs_a.cpu().detach()
        outputs_b_cpu = outputs_b.cpu().detach()
        change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5

        preds_a = torch.argmax(outputs_a_cpu, dim=1)
        preds_b = torch.argmax(outputs_b_cpu, dim=1)
        preds_a = (preds_a * change_mask.squeeze(1).long()).numpy()
        preds_b = (preds_b * change_mask.squeeze(1).long()).numpy()

        for pred_a, pred_b, label_a, label_b in zip(preds_a, preds_b, labels_a_np, labels_b_np):
            acc_a, _ = accuracy(pred_a, label_a)
            acc_b, _ = accuracy(pred_b, label_b)
            acc_meter.update((acc_a + acc_b) * 0.5)
            hist += get_hist(pred_a, label_a, args.num_classes)
            hist += get_hist(pred_b, label_b, args.num_classes)

    fscd, miou, sek = SCDD_eval_from_hist(hist)
    print(f"{time.time() - start:.1f}s Val loss: {val_loss.average():.4f} "
          f"Fscd: {fscd * 100:.2f} IoU: {miou * 100:.2f} "
          f"Sek: {sek * 100:.2f} Accuracy: {acc_meter.average() * 100:.2f}")
    return fscd, miou, sek, acc_meter.avg, val_loss.avg


def main():
    args = parse_args()
    print("=" * 80)
    print("Training Arguments:")
    for k, v in vars(args).items():
        print(f"{k:>24}: {v}")
    print("=" * 80)
    # set_visible_gpu(args.gpu)
    set_seed(args.seed, is_benchmark=True)

    log_dir = make_log_dir(args)
    print(f"Log dir: {log_dir}")
    spec = get_dataset_spec(args.data_name)
    args.num_classes = spec.num_classes

    net = build_model(args).cuda()
    print(f"Network {args.encoder} is initialized.")
    print("Number of network parameters: %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    print("Number of trainable parameters: %.1fM" % (sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6))

    train_loader, val_loader = build_loaders(args)
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    criterion_sc = SoftSemanticConsistency(reduction="mean", tau=args.tau).cuda()

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True
    )

    curr_epoch = 0
    best_fscd = 0.0
    if args.load_path is not None:
        net, optimizer, curr_epoch, best_fscd = resume_ckpt(net, optimizer, args.load_path)

    scaler = GradScaler()
    begin_time = time.time()
    all_iters = float(len(train_loader) * args.epochs)
    best_acc = 0.0
    best_loss = 1.0

    for epoch in range(curr_epoch, args.epochs):
        torch.cuda.empty_cache()
        seg_loss, bn_loss, sc_loss = train_one_epoch(
            args, train_loader, net, criterion, criterion_sc, optimizer, scaler, epoch, all_iters
        )
        fscd, miou, sek, acc, val_loss = validate(args, val_loader, net, criterion)

        if fscd > best_fscd:
            best_fscd = fscd
            best_acc = acc
            best_loss = val_loss
            ckpt_name = f"{args.encoder}_{epoch}e_mIoU{miou * 100:.2f}_Sek{sek * 100:.2f}_Fscd{fscd * 100:.2f}_OA{acc * 100:.2f}.pth"
            save_ckpt(os.path.join(log_dir, ckpt_name), epoch, net, optimizer, best_fscd, sek, miou)

        print(f"Epoch: {epoch} | Total time: {time.time() - begin_time:.1f}s | "
              f"Train seg {seg_loss:.4f} bn {bn_loss:.4f} sc {sc_loss:.4f} | "
              f"Best Val Fscd {best_fscd * 100:.2f} acc {best_acc * 100:.2f} loss {best_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()
