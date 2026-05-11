import os
import random
import math
import numbers
import collections
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class CDMToTensor(object):
    def __init__(self, normalize=True, target_type="uint8"):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, imga, imgb, lbla, lblb):
        if self.normalize:
            return (
                TF.to_tensor(imga),
                TF.to_tensor(imgb),
                torch.from_numpy(np.array(lbla, dtype=self.target_type)),
                torch.from_numpy(np.array(lblb, dtype=self.target_type)),
            )

        return (
            torch.from_numpy(np.array(imga, dtype=np.float32).transpose((2, 0, 1))),
            torch.from_numpy(np.array(imgb, dtype=np.float32).transpose((2, 0, 1))),
            torch.from_numpy(np.array(lbla, dtype=self.target_type)),
            torch.from_numpy(np.array(lblb, dtype=self.target_type)),
        )


class CDMNormalize(object):
    def __init__(self, mean_a, std_a, mean_b=None, std_b=None):
        self.mean_a = mean_a
        self.std_a = std_a
        self.mean_b = mean_b if mean_b is not None else mean_a
        self.std_b = std_b if std_b is not None else std_a

    def __call__(self, imga, imgb, lbla, lblb):
        return (
            TF.normalize(imga, self.mean_a, self.std_a, inplace=True),
            TF.normalize(imgb, self.mean_b, self.std_b, inplace=True),
            lbla,
            lblb,
        )


class CDMCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imga, imgb, lbla, lblb):
        for t in self.transforms:
            imga, imgb, lbla, lblb = t(imga, imgb, lbla, lblb)
        return imga, imgb, lbla, lblb


class CDMRandomFlipRotate(object):
    def __call__(self, imga, imgb, lbla, lblb):
        return rand_rot90_flip_SCD(imga, imgb, lbla, lblb)


class CDMColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non-negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}.")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if value[0] == value[1] == center:
            return None
        return value

    @staticmethod
    def get_trans(brightness, contrast, saturation, hue):
        transforms = []

        if brightness is not None:
            factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: TF.adjust_brightness(img, factor)))

        if contrast is not None:
            factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: TF.adjust_contrast(img, factor)))

        if saturation is not None:
            factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: TF.adjust_saturation(img, factor)))

        if hue is not None:
            factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: TF.adjust_hue(img, factor)))

        random.shuffle(transforms)
        return Compose(transforms)

    def __call__(self, imga, imgb, lbla, lblb):
        transform = self.get_trans(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(imga), transform(imgb), lbla, lblb


def rand_flip_SCD(img1, img2, label1, label2):
    r = random.random()

    if r < 0.25:
        return img1, img2, label1, label2
    elif r < 0.5:
        return (
            img1.transpose(Image.FLIP_TOP_BOTTOM),
            img2.transpose(Image.FLIP_TOP_BOTTOM),
            label1.transpose(Image.FLIP_TOP_BOTTOM),
            label2.transpose(Image.FLIP_TOP_BOTTOM),
        )
    elif r < 0.75:
        return (
            img1.transpose(Image.FLIP_LEFT_RIGHT),
            img2.transpose(Image.FLIP_LEFT_RIGHT),
            label1.transpose(Image.FLIP_LEFT_RIGHT),
            label2.transpose(Image.FLIP_LEFT_RIGHT),
        )
    else:
        return (
            img1.transpose(Image.ROTATE_180),
            img2.transpose(Image.ROTATE_180),
            label1.transpose(Image.ROTATE_180),
            label2.transpose(Image.ROTATE_180),
        )


def rand_rot90_SCD(img1, img2, label1, label2):
    if random.random() < 0.5:
        return img1, img2, label1, label2

    return (
        img1.transpose(Image.ROTATE_90),
        img2.transpose(Image.ROTATE_90),
        label1.transpose(Image.ROTATE_90),
        label2.transpose(Image.ROTATE_90),
    )


def rand_rot90_flip_SCD(img1, img2, label1, label2):
    img1, img2, label1, label2 = rand_rot90_SCD(img1, img2, label1, label2)
    return rand_flip_SCD(img1, img2, label1, label2)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    colormap: List[List[int]]
    classes: List[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "SECOND": DatasetSpec(
        name="SECOND",
        num_classes=7,
        colormap=[
            [255, 255, 255],
            [0, 128, 0],
            [128, 128, 128],
            [0, 255, 0],
            [0, 0, 255],
            [128, 0, 0],
            [255, 0, 0],
        ],
        classes=["unchanged", "low vegetation", "ground", "tree", "water", "building", "sports field"],
    ),
    "LandsatSCD": DatasetSpec(
        name="LandsatSCD",
        num_classes=5,
        colormap=[
            [255, 255, 255],
            [0, 155, 0],
            [255, 165, 0],
            [230, 30, 100],
            [0, 170, 240],
        ],
        classes=["unchanged", "farmland", "desert", "building", "water"],
    ),
}


NORM_PROFILES = {
    "imagenet": {
        "mean_a": (0.485, 0.456, 0.406),
        "std_a": (0.229, 0.224, 0.225),
        "mean_b": (0.485, 0.456, 0.406),
        "std_b": (0.229, 0.224, 0.225),
    },
    "pera": {
        "mean_a": (0.3585, 0.3741, 0.3155),
        "std_a": (0.1483, 0.1283, 0.1198),
        "mean_b": (0.3585, 0.3741, 0.3155),
        "std_b": (0.1483, 0.1283, 0.1198),
    },
}


ENCODER_TO_NORM = {
    "pera": "pera",
    "vmambaB": "imagenet",
    "resnet50": "imagenet",
    "swinV2L": "imagenet",
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_SPECS)}")
    return DATASET_SPECS[dataset_name]


def get_norm_profile(norm_profile: str, encoder: str):
    if norm_profile == "auto":
        norm_profile = ENCODER_TO_NORM.get(encoder, "imagenet")

    if norm_profile not in NORM_PROFILES:
        raise ValueError(f"Unknown norm profile: {norm_profile}")

    return NORM_PROFILES[norm_profile]


def index_to_color(mask: np.ndarray, dataset_name: str) -> np.ndarray:
    spec = get_dataset_spec(dataset_name)
    colormap = np.asarray(spec.colormap, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, spec.num_classes - 1)
    return colormap[mask]


def list_images(folder: str):
    return [n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS]


def pil_load(path: str):
    return Image.open(path)


class SCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        mode: str,
        dataset_name: str = "SECOND",
        encoder: str = "pera",
        norm_profile: str = "auto",
    ):
        self.root = root
        self.mode = mode
        self.dataset_name = dataset_name
        self.spec = get_dataset_spec(dataset_name)

        self.img_a_dir = os.path.join(root, mode, "im1")
        self.img_b_dir = os.path.join(root, mode, "im2")
        self.label_a_dir = os.path.join(root, mode, "label1")
        self.label_b_dir = os.path.join(root, mode, "label2")

        self.names = list_images(self.img_a_dir)

        norm = get_norm_profile(norm_profile, encoder)

        if mode == "train":
            self.transform = CDMCompose([
                CDMRandomFlipRotate(),
                CDMColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                CDMToTensor(),
                CDMNormalize(
                    mean_a=norm["mean_a"],
                    std_a=norm["std_a"],
                    mean_b=norm["mean_b"],
                    std_b=norm["std_b"],
                ),
            ])
        else:
            self.transform = CDMCompose([
                CDMToTensor(),
                CDMNormalize(
                    mean_a=norm["mean_a"],
                    std_a=norm["std_a"],
                    mean_b=norm["mean_b"],
                    std_b=norm["std_b"],
                ),
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_a = pil_load(os.path.join(self.img_a_dir, name))
        img_b = pil_load(os.path.join(self.img_b_dir, name))
        label_a = pil_load(os.path.join(self.label_a_dir, name))
        label_b = pil_load(os.path.join(self.label_b_dir, name))

        img_a, img_b, label_a, label_b = self.transform(img_a, img_b, label_a, label_b)
        return img_a, img_b, label_a, label_b, name