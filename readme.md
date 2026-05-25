# PerASCD: Plug-and-Play Semantic Change Detection Framework

<p align="center">
  <img src="assets/overview.png" width="85%">
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-12.1-76b900?logo=nvidia&logoColor=white"></a>
  <a href="#"><img src="https://img.shields.io/badge/Task-Semantic%20Change%20Detection-blue"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

> Official implementation of **PerASCD**, a semantic change detection framework with a plug-and-play encoder interface and a cascade gated decoder.

---

## News

- `[YYYY-MM-DD]` Code released.
- `[YYYY-MM-DD]` Pretrained models and checkpoints released.
- `[YYYY-MM-DD]` Paper accepted / submitted to `XXX`.

---

## Overview

Semantic Change Detection aims to identify changed regions between bi-temporal remote sensing images and predict their semantic categories before and after the change.

This repository provides:

- A unified training pipeline for semantic change detection.
- Multiple encoder backbones, including `PerA`, `VMamba-B`, `SwinV2-L`, and `ResNet50`.
- A Cascade Gated Decoder for multi-scale feature fusion.
- Support for `SECOND` and `LandsatSCD` datasets.
- Easy extension for custom encoders.

---

## Framework

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

The model follows a bi-temporal architecture:

```text
Image A ── Encoder ──┐
                     ├── Cascade Gated Decoder ── Semantic Map A
Image B ── Encoder ──┘                         └── Semantic Map B
                                                └── Binary Change Map
```

The decoder contains multi-scale fusion blocks and an optional refinement block:

```text
F4 → fuse F3 → fuse F2 → fuse F1 → optional refinement → prediction
```

---

## Project Structure

```text
PerASCD/
├── datasets/
│   ├── dataset.py              # Dataset and augmentation
│   └── RS_ST.py                # Legacy dataset implementation
├── models/
│   ├── common.py               # Shared decoder and SCDNet wrapper
│   ├── pera.py                 # PerA / DINOv2-G encoder
│   ├── vmamba.py               # VMamba-B encoder
│   ├── swin.py                 # SwinV2-L encoder
│   ├── resnet.py               # ResNet50 encoder
│   └── pera_layers/            # PerA / ViT-Adapter modules
├── utils/
│   ├── loss.py                 # Loss functions
│   ├── metrics.py              # Evaluation metrics
│   └── seed.py                 # Random seed control
├── train.py                    # Unified training script
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create environment

```bash
conda create -n perascd python=3.10 -y
conda activate perascd
```

### 2. Install PyTorch

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build custom CUDA ops

For PerA / ViT-Adapter, build `MultiScaleDeformableAttention`:

```bash
cd models/pera_layers/vit_adapter_layers/ops
sh make.sh
cd -
```

For RTX 4090 / 4090D, manually specify CUDA architecture when needed:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

Then rebuild the CUDA ops.

---

## Environment Notes

Recommended environment:

```text
Python        3.10
PyTorch       2.2.0 + cu121
torchvision   0.17.0
CUDA Toolkit  12.1
xFormers      0.0.24
mamba-ssm     2.2.4
```

Before running experiments, avoid system CUDA/cuDNN conflicts:

```bash
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
```

or directly:

```bash
LD_LIBRARY_PATH="" CUDA_VISIBLE_DEVICES=0 python train.py
```

---

## Dataset Preparation

### SECOND

Expected directory structure:

```text
SECOND/
├── train/
│   ├── im1/
│   ├── im2/
│   ├── label1/
│   └── label2/
└── test/
    ├── im1/
    ├── im2/
    ├── label1/
    └── label2/
```

Each label should be a single-channel `uint8` index map:

```text
0: unchanged / ignored
1~N: semantic classes
```

### LandsatSCD

Expected directory structure:

```text
LandsatSCD/
├── train/
│   ├── im1/
│   ├── im2/
│   ├── label1/
│   └── label2/
└── test/
    ├── im1/
    ├── im2/
    ├── label1/
    └── label2/
```

---

## Training

### Train PerA / DINOv2-G backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --encoder pera \
  --data-name SECOND \
  --data-path /path/to/SECOND \
  --input-size 448 \
  --output-size 512 \
  --train-batch-size 4 \
  --val-batch-size 4 \
  --grad-accum-steps 2 \
  --lr 0.1 \
  --drop-rate 0.3 \
  --norm-profile pera \
  --pretrained-path /path/to/pera_pretrained.params \
  --note pera_second
```

### Train VMamba-B backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --encoder vmambaB \
  --data-name SECOND \
  --data-path /path/to/SECOND \
  --input-size 512 \
  --output-size 512 \
  --train-batch-size 8 \
  --val-batch-size 8 \
  --grad-accum-steps 1 \
  --lr 0.1 \
  --drop-rate 0.3 \
  --norm-profile imagenet \
  --pretrained-path /path/to/vmamba_base.pth \
  --note vmamba_second
```

### Train SwinV2-L backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --encoder swinV2L \
  --data-name SECOND \
  --data-path /path/to/SECOND \
  --input-size 512 \
  --output-size 512 \
  --train-batch-size 4 \
  --val-batch-size 4 \
  --grad-accum-steps 2 \
  --lr 0.1 \
  --drop-rate 0.3 \
  --norm-profile imagenet \
  --pretrained-path /path/to/swinv2_large.safetensors \
  --note swin_second
```

### Train ResNet50 backbone

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --encoder resnet50 \
  --data-name SECOND \
  --data-path /path/to/SECOND \
  --input-size 512 \
  --output-size 512 \
  --train-batch-size 8 \
  --val-batch-size 8 \
  --grad-accum-steps 1 \
  --lr 0.1 \
  --drop-rate 0.3 \
  --norm-profile imagenet \
  --note resnet_second
```

---

## Evaluation

Validation is automatically performed after each training epoch.

The checkpoint name contains the main metrics:

```text
encoder_XXe_mIoUXX.XX_SekXX.XX_FscdXX.XX_OAXX.XX.pth
```

Example:

```text
pera_22e_mIoU86.59_Sek53.83_Fscd86.43_OA95.21.pth
```

---

## Results

### SECOND

| Method | Backbone | mIoU | Sek | Fscd | OA |
|---|---|---:|---:|---:|---:|
| ResNet50 + CG-Decoder | ResNet50 | XX.XX | XX.XX | XX.XX | XX.XX |
| SwinV2-L + CG-Decoder | SwinV2-L | XX.XX | XX.XX | XX.XX | XX.XX |
| VMamba-B + CG-Decoder | VMamba-B | XX.XX | XX.XX | XX.XX | XX.XX |
| **PerASCD** | PerA / DINOv2-G | **XX.XX** | **XX.XX** | **XX.XX** | **XX.XX** |

### LandsatSCD

| Method | Backbone | mIoU | Sek | Fscd | OA |
|---|---|---:|---:|---:|---:|
| ResNet50 + CG-Decoder | ResNet50 | XX.XX | XX.XX | XX.XX | XX.XX |
| SwinV2-L + CG-Decoder | SwinV2-L | XX.XX | XX.XX | XX.XX | XX.XX |
| VMamba-B + CG-Decoder | VMamba-B | XX.XX | XX.XX | XX.XX | XX.XX |
| **PerASCD** | PerA / DINOv2-G | **XX.XX** | **XX.XX** | **XX.XX** | **XX.XX** |

---

## Ablation Study

| Variant | mIoU | Sek | Fscd | OA |
|---|---:|---:|---:|---:|
| Full model | XX.XX | XX.XX | XX.XX | XX.XX |
| w/o CAGM | XX.XX | XX.XX | XX.XX | XX.XX |
| w/o SSCLoss | XX.XX | XX.XX | XX.XX | XX.XX |
| w/o refinement block | XX.XX | XX.XX | XX.XX | XX.XX |
| Decoder dim = 128 | XX.XX | XX.XX | XX.XX | XX.XX |
| Decoder dim = 256 | XX.XX | XX.XX | XX.XX | XX.XX |

---

## Custom Encoder

To add a new encoder, create a new file under `models/`, for example:

```text
models/my_encoder.py
```

Each encoder file should implement:

```python
def build_model(
    num_classes,
    input_size,
    output_size,
    drop_rate,
    pretrained_path=None,
    freeze_backbone=False,
):
    ...
    return model
```

The encoder should output a list of multi-scale features:

```python
features = [f1, f2, f3, f4]
```

where the resolution is usually:

```text
f1: 1/4
f2: 1/8
f3: 1/16
f4: 1/32
```

Then register it in `train.py`:

```python
ENCODER_REGISTRY = {
    "pera": "models.pera",
    "vmambaB": "models.vmamba",
    "resnet50": "models.resnet",
    "swinV2L": "models.swin",
    "myEncoder": "models.my_encoder",
}
```

---

## Reproducibility

To improve reproducibility, we recommend fixing the software environment, GPU, and random seed:

```bash
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
python train.py --seed 3701 ...
```

Exact bitwise reproducibility is not guaranteed due to CUDA, cuDNN, AMP, TensorCore, and multi-worker dataloader nondeterminism. We recommend reporting results from the same software and hardware environment.

---

## Troubleshooting

### 1. cuDNN library mismatch

Error:

```text
Could not load library libcudnn_cnn_train.so.8
undefined symbol ... libcudnn_cnn_infer.so.8
```

Solution:

```bash
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 python train.py ...
```

or:

```bash
LD_LIBRARY_PATH="" CUDA_VISIBLE_DEVICES=0 python train.py ...
```

### 2. MultiScaleDeformableAttention not found

Error:

```text
ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'
```

Solution:

```bash
cd models/pera_layers/vit_adapter_layers/ops
sh make.sh
```

### 3. CUDA architecture error

Error:

```text
no kernel image is available for execution on the device
```

For RTX 4090 / 4090D:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

Then rebuild CUDA ops.

### 4. xFormers not available

This warning may appear when xFormers is not installed:

```text
xFormers not available
```

Install a version compatible with your PyTorch and CUDA:

```bash
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121
```

---

## Citation

If this work is useful for your research, please consider citing:

```bibtex
@article{your2026perascd,
  title   = {PerASCD: A Plug-and-Play Framework for Semantic Change Detection},
  author  = {Your Name and Coauthors},
  journal = {XXX},
  year    = {2026}
}
```

---

## Acknowledgements

This project is built upon or inspired by:

- PyTorch
- timm
- DINOv2
- VMamba
- Swin Transformer
- ViT-Adapter
- SECOND dataset
- LandsatSCD dataset

---

## License

This project is released under the `MIT License`.

---

## Contact

For questions or suggestions, please contact:

```text
Your Name
your.email@example.com
```