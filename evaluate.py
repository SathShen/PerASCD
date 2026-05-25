import argparse
import importlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import SCDDataset, get_dataset_spec, index_to_color
from utils.metrics import accuracy, SCDD_eval_from_hist, AverageMeter, get_hist
from utils.seed import set_seed


ENCODER_REGISTRY = {
    "pera": "models.pera",
    "vmambaB": "models.vmamba",
    "resnet50": "models.resnet",
    "swinV2L": "models.swin",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SCD evaluation script.")

    parser.add_argument("--encoder", type=str, default="pera", choices=list(ENCODER_REGISTRY.keys()), help="Encoder backbone name.")
    parser.add_argument("--data-name", type=str, default="SECOND", help="Dataset name, e.g. SECOND or LandsatSCD.")
    parser.add_argument("--data-path", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split for evaluation, e.g. test or val.")
    parser.add_argument("--input-size", type=int, default=448, help="Input image size used by the model.")
    parser.add_argument("--output-size", type=int, default=512, help="Final output prediction size.")
    parser.add_argument("--norm-profile", type=str, default="auto", choices=["auto", "imagenet", "pera"], help="Normalization profile. Keep it consistent with training.")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Optional pretrained backbone path. Usually not needed when loading a full checkpoint.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone. Usually not needed for evaluation.")
    parser.add_argument("--drop-rate", type=float, default=0.3, help="Drop path / dropout rate used when building the model. Keep it consistent with training.")

    parser.add_argument("--load-path", type=str, required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for logs and predictions.")
    parser.add_argument("--save-preds", action="store_true", help="Save predA, predB, and change mask images.")
    parser.add_argument("--gen-conf-matrix", action="store_true", help="Save confusion matrices to eval.log.")

    parser.add_argument("--val-batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of dataloader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor for dataloader workers.")
    parser.add_argument("--seed", type=int, default=3701, help="Random seed.")

    return parser.parse_args()


def setup_logging(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(out_dir, "eval.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


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


def load_ckpt(net, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    net.load_state_dict(state_dict, strict=True)

    logging.info(f"Checkpoint loaded from {path}")
    torch.cuda.empty_cache()
    return net


def build_loader(args):
    val_set = SCDDataset(
        root=args.data_path,
        mode=args.split,
        dataset_name=args.data_name,
        encoder=args.encoder,
        norm_profile=args.norm_profile,
    )

    persistent_workers = args.num_workers > 0
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return val_loader


def colorize_index_map(index_map, dataset_name):
    return index_to_color(index_map, dataset_name)


@torch.no_grad()
def evaluate(args, val_loader, net, spec):
    net.eval()
    torch.cuda.empty_cache()

    start = time.time()
    acc_meter = AverageMeter()

    hist_total = np.zeros((args.num_classes, args.num_classes))
    hist_a = np.zeros((args.num_classes, args.num_classes))
    hist_b = np.zeros((args.num_classes, args.num_classes))

    pred_dir = os.path.join(args.out_dir, "pred")
    if args.save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    for data in tqdm(val_loader, ncols=80):
        imgs_a, imgs_b, labels_a, labels_b, imgs_id = data

        imgs_a = imgs_a.cuda(non_blocking=True).float()
        imgs_b = imgs_b.cuda(non_blocking=True).float()
        labels_a = labels_a.cuda(non_blocking=True).long()
        labels_b = labels_b.cuda(non_blocking=True).long()

        out_change, outputs_a, outputs_b = net(imgs_a, imgs_b)

        labels_a_np = labels_a.cpu().detach().numpy()
        labels_b_np = labels_b.cpu().detach().numpy()

        outputs_a_cpu = outputs_a.cpu().detach()
        outputs_b_cpu = outputs_b.cpu().detach()

        change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5

        preds_a = torch.argmax(outputs_a_cpu, dim=1)
        preds_b = torch.argmax(outputs_b_cpu, dim=1)

        preds_a = (preds_a * change_mask.squeeze(1).long()).numpy()
        preds_b = (preds_b * change_mask.squeeze(1).long()).numpy()

        change_mask_np = change_mask.numpy()

        for batch_idx in range(len(preds_a)):
            pred_a = preds_a[batch_idx]
            pred_b = preds_b[batch_idx]
            label_a = labels_a_np[batch_idx]
            label_b = labels_b_np[batch_idx]
            img_id = imgs_id[batch_idx]

            acc_a, _ = accuracy(pred_a, label_a)
            acc_b, _ = accuracy(pred_b, label_b)
            acc_meter.update((acc_a + acc_b) * 0.5)

            curr_hist_a = get_hist(pred_a, label_a, args.num_classes)
            curr_hist_b = get_hist(pred_b, label_b, args.num_classes)

            hist_a += curr_hist_a
            hist_b += curr_hist_b
            hist_total += curr_hist_a + curr_hist_b

            if args.save_preds:
                pred_a_color = colorize_index_map(pred_a, args.data_name)
                pred_b_color = colorize_index_map(pred_b, args.data_name)
                change_mask_img = (change_mask_np[batch_idx].squeeze() * 255).astype(np.uint8)

                io.imsave(os.path.join(pred_dir, f"{args.encoder}_{img_id}_predA.png"), pred_a_color)
                io.imsave(os.path.join(pred_dir, f"{args.encoder}_{img_id}_predB.png"), pred_b_color)
                io.imsave(os.path.join(pred_dir, f"{args.encoder}_{img_id}_changemask.png"), change_mask_img)

    fscd, miou, sek = SCDD_eval_from_hist(hist_total)
    elapsed = time.time() - start

    logging.info(
        f"{elapsed:.1f}s "
        f"Fscd: {fscd * 100:.2f} "
        f"mIoU: {miou * 100:.2f} "
        f"Sek: {sek * 100:.2f} "
        f"Accuracy: {acc_meter.average() * 100:.2f}"
    )

    if args.gen_conf_matrix:
        logging.info("Confusion Matrix for Image A:")
        logging.info(str(hist_a))
        logging.info("Confusion Matrix for Image B:")
        logging.info(str(hist_b))
        logging.info("Combined Confusion Matrix:")
        logging.info(str(hist_total))

    return fscd, miou, sek, acc_meter.average()


def main():
    args = parse_args()
    set_seed(args.seed, is_benchmark=True)

    spec = get_dataset_spec(args.data_name)
    args.num_classes = spec.num_classes

    if args.out_dir is None:
        timestamp = time.strftime("%y%m%d%H%M%S")
        args.out_dir = os.path.join("eval", args.data_name, args.encoder, timestamp)

    setup_logging(args.out_dir)

    logging.info("=" * 80)
    logging.info("Evaluation Arguments:")
    for k, v in vars(args).items():
        logging.info(f"{k:>24}: {v}")
    logging.info("=" * 80)

    net = build_model(args).cuda()

    logging.info(f"Network {args.encoder} initialized.")
    logging.info("Number of network parameters: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    logging.info("Number of trainable parameters: %.2fM" % (sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6))

    net = load_ckpt(net, args.load_path)

    val_loader = build_loader(args)

    fscd, miou, sek, acc = evaluate(args, val_loader, net, spec)

    logging.info("Evaluation finished.")
    logging.info(f"Final: Fscd={fscd * 100:.2f}, mIoU={miou * 100:.2f}, Sek={sek * 100:.2f}, OA={acc * 100:.2f}")


if __name__ == "__main__":
    main()