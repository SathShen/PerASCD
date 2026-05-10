import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用GPU0
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/home/sht/times.ttf'  # <--- 修改为你的实际路径
fm.fontManager.addfont(font_path)
# 3. 获取该字体的内部名称 (Matplotlib 需要知道它的注册名，通常是 'Times New Roman')
prop = fm.FontProperties(fname=font_path)
custom_font_name = prop.get_name() 
# 4. 更新全局 rcParams 配置
plt.rcParams['font.family'] = custom_font_name  # 将默认字体家族设置为刚才加载的字体
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为 Times New Roman
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题

# --- 字体大小设置 ---
plt.rcParams['font.size'] = 14          # 全局基础字体大小
plt.rcParams['axes.titlesize'] = 20     # 子图标题 (Title) 大小
plt.rcParams['axes.labelsize'] = 14     # 轴标签 (Label) 大小
plt.rcParams['xtick.labelsize'] = 14    # X轴刻度大小
plt.rcParams['ytick.labelsize'] = 14    # Y轴刻度大小
plt.rcParams['legend.fontsize'] = 14    # 图例字体大小
# -----------------------------------------------------------

# 导入你的项目依赖
from utils.seed import set_seed
from datasets import RS_ST as RS
from models.PerAChain import PerASCD

# 配置参数 (请根据你的实际路径修改)
args = {
    'gpu': True,
    'seed': 3701,
    'arch': 'ViT-G/16/1024', # 或 'ViT-B/16'，需与权重匹配
    'droppath': 0.0,
    # 你的权重文件路径
    'load_path': '/data2/sht/Outputs/SCD/runs/SECOND/PerAChain/260128115444_vitg01min0Clip15LsscTau001/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth', 
    'vis_dir': '/data2/sht/Outputs/vis_cagm_results', # 结果保存路径
    'val_batch_size': 1,
    'num_workers': 1,
    'input_size': 448, # 图像输入尺寸
    'pretrained_pera_path': None, # 预测时不需要加载预训练权重，只需要加载ckpt
    'is_distilled': False
}

if not os.path.exists(args['vis_dir']):
    os.makedirs(args['vis_dir'])

# -----------------------------------------------------------
# Hook 定义：用于截取中间层输出
# -----------------------------------------------------------
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # output 是 CAGM 的返回值，即 change_map (B, 2, H, W)
        activation[name] = output.detach()
    return hook

# -----------------------------------------------------------
# 辅助函数：反归一化图像以便可视化
# -----------------------------------------------------------
def denormalize(img_tensor):
    """
    假设使用了常见的 ImageNet 均值和方差，如果你的数据集不同请修改
    img_tensor: (C, H, W)
    """
    mean = np.array([0.3585, 0.3741, 0.3155])
    std = np.array([0.1483, 0.1283, 0.1198])
    
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def colorize_mask(mask):
    if hasattr(RS, 'Index2Color'):
        return RS.Index2Color(mask)
    else:
        # Fallback if utils not available
        return mask

# -----------------------------------------------------------
# 主函数
# -----------------------------------------------------------
def main():
    set_seed(args['seed'], is_benchmark=True)
    device = torch.device('cuda' if args['gpu'] and torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型
    print(f"Initializing Network: {args['arch']}...")
    net = PerASCD(in_channels=3, 
                  num_classes=RS.num_classes, 
                  input_size=args['input_size'],
                  output_size=512,
                  arch=args['arch'],
                  droppath=args['droppath'],
                  pretrained_pera_path=None, 
                  is_distilled_pera=False,
                  is_freeze_backbone=False).to(device)

    # 2. 加载权重
    if os.path.isfile(args['load_path']):
        print(f"Loading checkpoint from {args['load_path']}")
        checkpoint = torch.load(args['load_path'], weights_only=False)
        # 处理可能的 key 不匹配问题
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        net.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args['load_path']}")

    net.eval()

    # 3. 注册 Hook 到每一个 Block 的 CAGM 模块
    # net.decoder.blocks 是一个 nn.ModuleList
    for i, block in enumerate(net.decoder.blocks):
        # 这里的 block.cagm 对应代码中的 self.cagm = ChangeAwareGatingModule(...)
        block.cagm.register_forward_hook(get_activation(f'block_{i}'))
        print(f"Registered hook for decoder block {i} CAGM")

    # 4. 准备数据
    val_set = RS.DataPerAAUG('test') # 确保使用验证集
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False, 
                            num_workers=args['num_workers'], pin_memory=True)

    print(f"Start visualizing... saving to {args['vis_dir']}")

    # 5. 推理循环
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            imgs_A, imgs_B, labels_A, labels_B, imgs_id = data
            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()
            
            # 清空上一轮的 activation (虽然字典会自动覆盖，但为了安全)
            activation.clear()

            # 前向传播
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            
            # 获取当前图片ID
            img_name = imgs_id[0]

            # --- 数据后处理 ---
            # 1. 变化检测结果 (Binary)
            pred_change_mask = torch.sigmoid(out_change[0, 0]) > 0.5 # (H, W) Bool
            
            # 2. 语义分割结果 (Indices)
            pred_A_idx = torch.argmax(outputs_A, dim=1)[0] # (H, W)
            pred_B_idx = torch.argmax(outputs_B, dim=1)[0] # (H, W)
            
            # 3. 将语义结果与变化检测结果结合 (过滤掉未变化区域)
            # 在 SCD 任务中，通常只关心变化区域的语义，或认为未变化区域语义为背景(0)
            pred_A_idx = pred_A_idx * pred_change_mask.long()
            pred_B_idx = pred_B_idx * pred_change_mask.long()

            # 4. 转换为 RGB 可视化图
            pred_A_vis = colorize_mask(pred_A_idx.cpu().numpy())
            pred_B_vis = colorize_mask(pred_B_idx.cpu().numpy())
            gt_A_vis   = colorize_mask(labels_A[0].numpy())
            gt_B_vis   = colorize_mask(labels_B[0].numpy())
            gt_mask_vis = ((labels_A[0] != labels_B[0]).cpu().numpy()).astype(np.uint8) * 255  # to binary mask
            change_vis = pred_change_mask.cpu().numpy()

            # --- 绘图逻辑 ---
            num_blocks = len(net.decoder.blocks)
            # 布局规划：
            # Row 1: [Img A] [GT A] [Pred A] [Pred Change]
            # Row 2: [Img B] [GT B] [Pred B] [Blank]
            # Row 3+: [High W] [Low W] [Shape Info] [Blank]
            
            fig_rows = 4
            fig_cols = max(4, num_blocks) 
            plt.figure(figsize=(16, 4 * fig_rows))

            # === Row 1: Image A Context ===
            # 1. Img A
            plt.subplot(fig_rows, fig_cols, 1)
            plt.imshow(denormalize(imgs_A[0]))
            # plt.title(f"Image A: {img_name}")
            plt.title(f"Image A")
            plt.axis('off')
            # 2. GT A
            plt.subplot(fig_rows, fig_cols, 2)
            plt.imshow(gt_A_vis)
            plt.title("GT Label A")
            plt.axis('off')
            # 6. GT B
            plt.subplot(fig_rows, fig_cols, 3)
            plt.imshow(gt_B_vis)
            plt.title("GT Label B")
            plt.axis('off')
            # 6. GT Mask
            plt.subplot(fig_rows, fig_cols, 4)
            plt.imshow(gt_mask_vis, cmap='gray', vmin=0, vmax=255)
            plt.title("GT Mask")
            plt.axis('off')
            # === Row 2: Image B Context ===
            # 5. Img B
            plt.subplot(fig_rows, fig_cols, 5)
            plt.imshow(denormalize(imgs_B[0]))
            plt.title("Image B")
            plt.axis('off')
            # 3. Pred A
            plt.subplot(fig_rows, fig_cols, 6)
            plt.imshow(pred_A_vis)
            plt.title("Pred Label A")
            plt.axis('off')
            # 7. Pred B
            plt.subplot(fig_rows, fig_cols, 7)
            plt.imshow(pred_B_vis)
            plt.title("Pred Label B")
            plt.axis('off')
            # 4. Pred Change
            plt.subplot(fig_rows, fig_cols, 8)
            plt.imshow(change_vis, cmap='gray')
            plt.title("Pred Change Mask")
            plt.axis('off')

            # === Row 3+: Block CAGM Outputs ===
            for b_idx in range(num_blocks):
                layer_name = f'block_{b_idx}'
                if layer_name not in activation:
                    continue
                cagm_out = activation[layer_name] # [B, 2, h, w]
                
                # 上采样到原图尺寸
                cagm_out_resized = F.interpolate(cagm_out, size=(args['input_size'], args['input_size']), mode='bilinear', align_corners=False)
                w_high = cagm_out_resized[0, 0].cpu().numpy()
                w_low  = cagm_out_resized[0, 1].cpu().numpy()
                
                high_idx = 3 * fig_cols
                low_idx = 2 * fig_cols

                # 1. Low Weight
                plt.subplot(fig_rows, fig_cols, low_idx + b_idx + 1)
                plt.imshow(w_low, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"Block {b_idx}: Low Feat Weight")
                plt.axis('off')

                # 2. High Weight
                plt.subplot(fig_rows, fig_cols, high_idx + b_idx + 1)
                plt.imshow(w_high, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"Block {b_idx}: High Feat Weight")
                plt.axis('off')
                
                # # 3. Shape Info (Replacing Diff)
                # plt.subplot(fig_rows, fig_cols, base_idx + 3)
                # plt.text(0.5, 0.5, f"Feature Shape:\n{tuple(cagm_out.shape)}", ha='center', va='center', fontsize=12)
                # plt.title(f"Block {b_idx} Info")
                # plt.axis('off')

                # # 4. Blank
                # plt.subplot(fig_rows, fig_cols, base_idx + 4)
                # plt.axis('off')

            plt.tight_layout()
            save_name = os.path.join(args['vis_dir'], f"{img_name}_cagm_vis.png")
            plt.savefig(save_name, dpi=150)
            plt.close()

            # 仅可视化前 10 张图片
            # if i >= 10:
            #     print("Visualized 10 images, stopping.")
            #     break

if __name__ == '__main__':
    main()