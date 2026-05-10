import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用GPU0
import math
import time
import copy
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

working_path = os.path.dirname(os.path.abspath(__file__))
# ce loss
from torch.nn import CrossEntropyLoss as CrossEntropyLoss2d
from torch.nn import BCEWithLogitsLoss
# focal loss
from utils.utils import accuracy, SCDD_eval_from_hist, AverageMeter, get_hist

# Data and model choose
###############################################
from datasets import RS_ST as RS
#from models.TED import TED as Net
from models.HRSCD import HRSCD2
NET_NAME = 'HRSCD2'
DATA_NAME = 'SECOND'
###############################################
# Training options
###############################################
args = {
    'train_batch_size': 8,
    'val_batch_size': 8,
    'lr': 0.1,
    'gpu': True,
    'epochs': 50,
    'num_workers': 2,
    'lr_decay_power': 1.5,
    'psd_train': True,
    'psd_TTA': True,
    'vis_psd': True,
    'psd_init_Fscd': 0.6,
    'print_freq': 50,
    'predict_step': 5,
    'pseudo_thred': 0.6,
    'log_dir': f'/data2/sht/Outputs/SCD/runs/{DATA_NAME}/{NET_NAME}/exp_{int(time.time())}',
    'load_path': None
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
writer = SummaryWriter(args['log_dir'])

class AverageThred(object):
    def __init__(self, num_classes):        
        self.threds = np.ones((num_classes), dtype=float)*args['pseudo_thred']
        self.count = np.ones((num_classes), dtype=int)
        self.sum = self.threds*self.count

    def update(self, threds, count):
        self.count += np.array(count, dtype=int)
        self.sum += threds*count
        self.threds = self.sum/self.count

    def value(self):
        return np.clip(self.threds, 0.5, 0.9)

def calc_conf(prob, conf_thred):
    b, c, h, w = prob.size()
    conf, index = torch.max(prob, dim=1)
    index_onehot = F.one_hot(index.long(), num_classes=RS.num_classes).permute((0,3,1,2))
    masked_prob = index_onehot*prob
    threds, len_c = np.zeros(c), np.zeros(c)
    for idx in range(c):
        masked_prob_i = torch.flatten(masked_prob[:, idx])
        masked_prob_i = masked_prob_i[masked_prob_i.nonzero()]
        len = masked_prob_i.size(0)
        
        if len>0:
            conf_thred_i = np.percentile(masked_prob_i.cpu().numpy().flatten(), 100*args['pseudo_thred'])
            threds[idx] = conf_thred_i
            len_c[idx] = len
        else:
            threds[idx] = args['pseudo_thred']
            len_c[idx] = 0
        
    conf_thred.update(threds, len_c)
    threds = torch.from_numpy(conf_thred.value()).unsqueeze(1).unsqueeze(2).cuda()
    thred_onehot = index_onehot*threds
    thredmap, _ = torch.max(thred_onehot, dim=1)
    conf = torch.ge(conf, thredmap)
    return conf, index

def save_ckpt(path, epoch, net, optimizer, bestFscdV, Sek_v, mIoU_v):
    state = {
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'Fscd': bestFscdV,
        'Sek': Sek_v,
        'mIoU': mIoU_v
    }
    torch.save(state, path)
    print('Checkpoint saved to %s' % path)


def resume_ckpt(net, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    net.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Checkpoint loaded from %s' % path)
    return net, optimizer, checkpoint['epoch'] + 1, checkpoint['Fscd']
    # return net, optimizer, checkpoint['epoch'] + 1, 0


def main():
    net = HRSCD2(3, num_classes=RS.num_classes).cuda()

    train_set = RS.Data('train', random_flip=True, random_swap=False)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True, num_workers=args['num_workers'], 
                              pin_memory=True, drop_last=True, persistent_workers=True)
    val_set = RS.Data('test')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False, num_workers=args['num_workers'], 
                            pin_memory=True, drop_last=False, persistent_workers=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=5e-4, momentum=0.9, nesterov=True)
    curr_epoch = 0
    bestFscdV = 0.0
    if args['load_path'] is not None:
        net, optimizer, curr_epoch, bestFscdV = resume_ckpt(net, optimizer, args['load_path'])
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=(0.9, 0.999))

    train(train_loader, net, optimizer, val_loader, curr_epoch, bestFscdV)
    writer.close()
    print('Training finished.')

def train(train_loader, net, optimizer, val_loader, curr_epoch=0, bestFscdV=0.0):

    scaler = GradScaler()
    bestaccT = 0
    bestaccV = 0.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    
    while True:
        torch.cuda.empty_cache()
        net.train()
        # freeze_model(net.FCN)
        start = time.time()
        acc_meter = AverageMeter()
        train_lcm = AverageMeter()
        criterion_Lcm = CrossEntropyLoss2d().cuda()



        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(tqdm(train_loader)):
            running_iter = curr_iter + i + 1
            iter_ratio = running_iter/all_iters
            adjust_lr(optimizer, iter_ratio)
            imgs_A, imgs_B, labels_A, labels_B, imgs_id = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_cd = (labels_A > 0).unsqueeze(1).cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()
                label = net.double_label_2_single_label(labels_A, labels_B)

            with autocast(device_type='cuda', dtype=torch.float16):
                out = net(imgs_A, imgs_B)
                        
                loss = criterion_Lcm(out, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


            out_change, outputs_A, outputs_B = net.post_process(out)
            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_lcm.update(loss.cpu().detach().numpy())

            writer.add_scalar('train_lcm', train_lcm.val, running_iter)
            writer.add_scalar('train accuracy', acc_meter.val*100, running_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)
            
            curr_time = time.time() - start
            # if (i + 1) % args['print_freq'] == 0:
            #     print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
            #         curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
            #         train_seg_loss.val, train_bn_loss.val, acc_meter.val * 100))  # sc_loss %.4f, train_sc_loss.val, 

        Fscd_v, mIoU_v, Sek_v, acc_v = validate(val_loader, net, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if Fscd_v>bestFscdV:
            bestFscdV=Fscd_v
            bestaccV=acc_v
            save_path = os.path.join(args['log_dir'], NET_NAME+'_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth'\
                %(curr_epoch, mIoU_v*100, Sek_v*100, Fscd_v*100, acc_v*100))
            save_ckpt(save_path, curr_epoch, net, optimizer, bestFscdV, Sek_v, mIoU_v)
        print('Epoch: %d Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f' 
              %(curr_epoch, time.time()-begin_time, bestaccT*100, bestFscdV*100, bestaccV*100))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    acc_meter = AverageMeter()

    hist = np.zeros((RS.num_classes, RS.num_classes))

    for vi, data in enumerate(tqdm(val_loader)):
        imgs_A, imgs_B, labels_A, labels_B, imgs_id = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            out = net(imgs_A, imgs_B)
            out_change, outputs_A, outputs_B = net.post_process(out)

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()

        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)
            hist += get_hist(preds_A, labels_A, RS.num_classes)
            hist += get_hist(preds_B, labels_B, RS.num_classes)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['log_dir'], NET_NAME + '_A.png'), pred_A_color)
            io.imsave(os.path.join(args['log_dir'], NET_NAME + '_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Sek = SCDD_eval_from_hist(hist)

    curr_time = time.time() - start
    print('%.1fs  Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(curr_time, Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))

    writer.add_scalar('val_Fscd', Fscd*100, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average()*100, curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, iter_ratio, init_lr=args['lr']):
    #scale_running_lr = math.sin((iter_ratio)*math.pi/2)
    scale_running_lr = ((1. - iter_ratio) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr



if __name__ == '__main__':
    main()
