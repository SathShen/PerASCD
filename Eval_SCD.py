import os
import time
import numpy as np
from skimage import io
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from utils.seed import set_seed
import time
import logging
import warnings
import argparse
from models.Encoders import build_net
warnings.filterwarnings('ignore', category=UserWarning)

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_from_hist, AverageMeter, get_hist

# Data and model choose
###############################################
from datasets import RS_ST as RS
# from models.PerAChain import PerASCD
# from models.SCanNet import SCanNet
# from models.TED import TED
# from models.SSCDl import SSCDl
# from models.BiSRNet import BiSRNet
# from models.HRSCD import HRSCD2, HRSCD3, HRSCD4

NET_NAME = 'resnet50'
DATA_NAME = 'SECOND'
# DATA_NAME = 'LandsatSCD'
NOTE=""
###############################################
# Evaluation options
###############################################
# args = {
#     'val_batch_size': 1,
#     'gpu': True,
#     'num_workers': 16,
#     'seed': 3701,
#     'arch': 'ViT-G/16/1024',
#     'out_dir': f'/data2/sht/Outputs/SCD/eval/{DATA_NAME}/{NET_NAME}/{NOTE}{time.strftime("%y%m%d%H%M%S")}',
#     'load_path': "/data2/sht/Outputs/SCD/runs/SECOND/BiSRNet/exp_1768802902/BiSRNet_49e_mIoU73.29_Sek23.05_Fscd63.10_OA87.67.pth",  # Specify the path to the trained model checkpoint
#     'gen_conf_matrix': True,  # Whether to generate and print confusion matrices for A and B (text output)
#     'save_preds': True,  # Whether to save predA, predB, and changemask as images
#     'is_distilled': False,
#     'dataset_split': 'test',  # 'val' or 'test' if available in RS.DataPerAAUG
# }
###############################################



def load_ckpt(net, path):
    checkpoint = torch.load(path, weights_only=False)
    net.load_state_dict(checkpoint['model'], strict=True)
    logging.info(f"Checkpoint loaded from {path}.")
    torch.cuda.empty_cache()
    return net



def main():
    parser = argparse.ArgumentParser(description='resnet50+CGdecoder evaluation')
#     'num_workers': 16,
#     'seed': 3701,
#     'arch': 'ViT-G/16/1024',
#     'out_dir': f'/data2/sht/Outputs/SCD/eval/{DATA_NAME}/{NET_NAME}/{NOTE}{time.strftime("%y%m%d%H%M%S")}',
#     'load_path': "/data2/sht/Outputs/SCD/runs/SECOND/BiSRNet/exp_1768802902/BiSRNet_49e_mIoU73.29_Sek23.05_Fscd63.10_OA87.67.pth",  # Specify the path to the trained model checkpoint
#     'gen_conf_matrix': True,  # Whether to generate and print confusion matrices for A and B (text output)
#     'save_preds': True,  # Whether to save predA, predB, and changemask as images
#     'is_distilled': False,
#     'dataset_split': 'test',  # 'val' or 'test' if available in RS.DataPerAAUG
    # 2. 添加参数
    parser.add_argument('--test_path', '-t', type=str, help='Your test dataset dir' )
    parser.add_argument('--out_dir', '-o', type=str, help='Your output dir')          # 位置参数
    parser.add_argument('--load_path', '-l', type=str, help='Path to the model checkpoint') # 可选参数
    parser.add_argument('--save_preds', '-s', action='store_true', help='Whether to save predictions') # 可选参数
    parser.add_argument('--gen_conf_matrix', '-c', action='store_true', help='Whether to generate confusion matrices') # 可选参数
    parser.add_argument('--num_workers', '-n', type=int, default=4, help='Number of data loading workers') # 可选参数

    # 3. 解析参数
    args = parser.parse_args()

    if not os.path.exists(args.out_dir): os.makedirs(f"{args.out_dir}/pred")
    if os.path.exists(args.load_path) is None:
        raise ValueError("load_path must be specified for evaluation.")

    # Set up logging
    logging.basicConfig(filename=f'{args.out_dir}/eval.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info(f"Evaluation started with args: {args}")


    set_seed(3701, is_benchmark=True)
    # net = PerASCD(in_channels=3, 
    #               num_classes=RS.num_classes, 
    #               input_size=448,
    #               output_size=512,
    #               arch=args['arch'],
    #               droppath=0.0,  # No droppath during eval
    #               pretrained_pera_path=None,
    #               is_distilled_pera=args['is_distilled'],
    #               is_freeze_backbone=False).cuda()
    net = build_net(NET_NAME, RS.num_classes, output_size=512, drop_rate=0)
    net = net.cuda()

    logging.info(f"Network {NET_NAME} initialized.")
    logging.info(f"Number of network parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    if args.load_path is not None:
        net = load_ckpt(net, args.load_path)
    else:
        raise ValueError("load_path must be specified for evaluation.")
    

    val_set = RS.DataPerAAUG('test', path=args.test_path)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, 
                            pin_memory=True, drop_last=False, persistent_workers=True, prefetch_factor=4)

    # criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    evaluate(val_loader, net, args)
    logging.info('Evaluation finished.')

def evaluate(val_loader, net, args):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()
    acc_meter = AverageMeter()

    hist_total = np.zeros((RS.num_classes, RS.num_classes))
    hist_A = np.zeros((RS.num_classes, RS.num_classes))
    hist_B = np.zeros((RS.num_classes, RS.num_classes))

    for vi, data in enumerate(tqdm(val_loader, ncols=80)):
        imgs_A, imgs_B, labels_A, labels_B, imgs_id = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()

        with torch.no_grad():
            if NET_NAME == "HRSCD2":
                out = net(imgs_A, imgs_B)
                out_change, outputs_A, outputs_B = net.post_process(out)
            else:
                out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

        labels_A_np = labels_A.cpu().detach().numpy()
        labels_B_np = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        out_change = out_change.cpu().detach()
        change_mask = (F.sigmoid(out_change) > 0.5).numpy()
        preds_A = torch.argmax(outputs_A, dim=1).numpy()
        preds_B = torch.argmax(outputs_B, dim=1).numpy()
        # Apply change mask only to changed areas
        preds_A = preds_A * change_mask.squeeze()
        preds_B = preds_B * change_mask.squeeze()

        for batch_idx in range(len(preds_A)):
            pred_A = preds_A[batch_idx]
            pred_B = preds_B[batch_idx]
            label_A = labels_A_np[batch_idx]
            label_B = labels_B_np[batch_idx]
            img_id = imgs_id[batch_idx]

            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

            curr_hist_A = get_hist(pred_A, label_A, RS.num_classes)
            curr_hist_B = get_hist(pred_B, label_B, RS.num_classes)
            # 改为每次只加当前图像的增量
            hist_A += curr_hist_A
            hist_B += curr_hist_B
            hist_total += curr_hist_A + curr_hist_B  # Combined for total metrics

            if args.save_preds:
                pred_A_color = RS.Index2Color(pred_A)
                pred_B_color = RS.Index2Color(pred_B)
                change_mask_img = (change_mask[batch_idx].squeeze() * 255).astype(np.uint8)
                io.imsave(os.path.join(args.out_dir, 'pred', f"{NET_NAME}_{img_id}_predA.png"), pred_A_color)
                io.imsave(os.path.join(args.out_dir, 'pred', f"{NET_NAME}_{img_id}_predB.png"), pred_B_color)
                io.imsave(os.path.join(args.out_dir, 'pred', f"{NET_NAME}_{img_id}_changemask.png"), change_mask_img)

    Fscd, mIoU, Sek = SCDD_eval_from_hist(hist_total)

    curr_time = time.time() - start
    log_msg = f"{curr_time:.1f}s Fscd: {Fscd*100:.2f} mIoU: {mIoU*100:.2f} Sek: {Sek*100:.2f} Accuracy: {acc_meter.average()*100:.2f}"
    logging.info(log_msg)

    if args.gen_conf_matrix:
        logging.info("Confusion Matrix for Image A:")
        logging.info(str(hist_A))
        logging.info("Confusion Matrix for Image B:")
        logging.info(str(hist_B))
        logging.info("Combined Confusion Matrix:")
        logging.info(str(hist_total))

    return Fscd, mIoU, Sek, acc_meter.average()

if __name__ == '__main__':
    main()