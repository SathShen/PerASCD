import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF



#--------------
from torchvision.transforms import functional as TF
import PIL
import math
from PIL import Image
from torchvision.transforms import InterpolationMode
import collections
import numbers

class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class CDMToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, imga, imgb, lbla, lblb):
        if self.normalize:
            return TF.to_tensor(imga),\
                    TF.to_tensor(imgb),\
                    torch.from_numpy(np.array(lbla, dtype=self.target_type)),\
                    torch.from_numpy(np.array(lblb, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(imga, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(imgb, dtype=np.float32).transpose((2, 0, 1))),\
                   torch.from_numpy(np.array(lbla, dtype=self.target_type)),\
                   torch.from_numpy(np.array(lblb, dtype=self.target_type))

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class CDMNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imga, imgb, lbla, lblb):
        return TF.normalize(imga, self.mean, self.std, inplace=True),\
               TF.normalize(imgb, self.mean, self.std, inplace=True),\
               lbla, lblb

    
class CDMCompose(object):
    """Composes several transforms together.
    Args: transforms (list of ``Transform`` objects): list of transforms to compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imga, imgb, lbla, lblb):
        for t in self.transforms:
            imga, imgb, lbla, lblb = t(imga, imgb, lbla, lblb)
        return imga, imgb, lbla, lblb
    
    def append(self, transform):
        self.transforms.append(transform)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    
class CDMRandomFlipRotate(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        pass

    def __call__(self, imga, imgb, lbla, lblb):
        imga, imgb, lbla, lblb = rand_rot90_flip_SCD(imga, imgb, lbla, lblb)
        return imga, imgb, lbla, lblb
    

class CDMRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max)")
            sys.exit()

        self.interpolation = interpolation

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop."""
        if isinstance(img, torch.Tensor):
            _, height, width = img.shape
        elif isinstance(img, PIL.Image.Image):
            width, height = img.size
        else:
            raise TypeError("Unexpected type {}".format(type(img)))
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, imga, imgb, lbla, lblb):
        i, j, h, w = self.get_params(imga, self.scale, self.ratio)
        return TF.resized_crop(imga, i, j, h, w, self.size, self.interpolation, antialias=True),\
               TF.resized_crop(imgb, i, j, h, w, self.size, self.interpolation, antialias=True),\
               TF.resized_crop(lbla, i, j, h, w, self.size, InterpolationMode.NEAREST),\
               TF.resized_crop(lblb, i, j, h, w, self.size, InterpolationMode.NEAREST)


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


class CDMColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_trans(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: TF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, imga, imgb, lbla, lblb):
        transform = self.get_trans(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        transed_imga = transform(imga)
        transed_imgb = transform(imgb)
        return transed_imga, transed_imgb, lbla, lblb

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


def rand_flip_SCD(img1, img2, label1, label2):
    r = random.random()

    if r < 0.25:
        # no flip
        return img1, img2, label1, label2
    elif r < 0.5:
        # vertical flip
        return (
            img1.transpose(Image.FLIP_TOP_BOTTOM),
            img2.transpose(Image.FLIP_TOP_BOTTOM),
            label1.transpose(Image.FLIP_TOP_BOTTOM),
            label2.transpose(Image.FLIP_TOP_BOTTOM),
        )
    elif r < 0.75:
        # horizontal flip
        return (
            img1.transpose(Image.FLIP_LEFT_RIGHT),
            img2.transpose(Image.FLIP_LEFT_RIGHT),
            label1.transpose(Image.FLIP_LEFT_RIGHT),
            label2.transpose(Image.FLIP_LEFT_RIGHT),
        )
    else:
        # both flips (same as 180° rotation)
        return (
            img1.transpose(Image.ROTATE_180),
            img2.transpose(Image.ROTATE_180),
            label1.transpose(Image.ROTATE_180),
            label2.transpose(Image.ROTATE_180),
        )


def rand_rot90_SCD(img1, img2, label1, label2):
    r = random.random()
    if r < 0.5:
        # no rotation
        return img1, img2, label1, label2
    else:
        # 90° rotation
        return (
            img1.transpose(Image.ROTATE_90),
            img2.transpose(Image.ROTATE_90),
            label1.transpose(Image.ROTATE_90),
            label2.transpose(Image.ROTATE_90),
        )


def rand_rot90_flip_SCD(img1, img2, label1, label2):
    img1, img2, label1, label2 = rand_rot90_SCD(img1, img2, label1, label2)
    return rand_flip_SCD(img1, img2, label1, label2)
#-----------------------------------------


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
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "pera": {
        "mean": (0.3585, 0.3741, 0.3155),
        "std": (0.1483, 0.1283, 0.1198),
    },
}


ENCODER_TO_NORM = {
    "pera": "pera",
    "vmambaB": "imagenet",
    "resnet50": "imagenet",
    "swinV2L": "imagenet",
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_SPECS)}")
    return DATASET_SPECS[dataset_name]


def get_norm_profile(norm_profile: str, encoder: Optional[str] = None):
    if norm_profile == "auto":
        if encoder is None:
            raise ValueError("norm_profile='auto' requires encoder name.")
        norm_profile = ENCODER_TO_NORM.get(encoder, "imagenet")

    if norm_profile not in NORM_PROFILES:
        raise ValueError(f"Unknown norm profile '{norm_profile}'. Available: {list(NORM_PROFILES)} plus 'auto'.")
    return NORM_PROFILES[norm_profile]


def index_to_color(mask: np.ndarray, dataset_name: str) -> np.ndarray:
    spec = get_dataset_spec(dataset_name)
    colormap = np.asarray(spec.colormap, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.int64)
    mask = np.clip(mask, 0, spec.num_classes - 1)
    return colormap[mask]


def is_image_file(img_ext):
    return img_ext == 'tif' or img_ext == 'tiff' or img_ext == 'png' or img_ext == 'jpg' or img_ext == 'jpeg' or img_ext == 'JPEG'

def pil_load(path: str) -> Image.Image:
    if is_image_file(path.split('.')[-1]):
        img = Image.open(path)
        return img


class SCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        mode: str,
        dataset_name: str = "SECOND",
        encoder: Optional[str] = None,
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

        self.imgs_name_list = os.listdir(self.img_a_dir)
        if len(self.imgs_name_list) == 0:
            raise RuntimeError(f"No images found in {self.img_a_dir}")

        norm = get_norm_profile(norm_profile, encoder)
        print(norm)

        if mode == "train":
            self.transform = CDMCompose([
                CDMRandomFlipRotate(),
                CDMColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.1,
                ),
                CDMToTensor(),
                CDMNormalize(
                    mean=norm["mean"],
                    std=norm["std"],
                ),
            ])
        else:
            self.transform = CDMCompose([
                CDMToTensor(),
                CDMNormalize(
                    mean=norm["mean"],
                    std=norm["std"],
                ),
            ])

    def __len__(self):
        return len(self.imgs_name_list)

    def __getitem__(self, idx):
        name = os.path.basename(self.imgs_name_list[idx])

        img_a = pil_load(os.path.join(self.img_a_dir, name))
        img_b = pil_load(os.path.join(self.img_b_dir, name))
        label_a = pil_load(os.path.join(self.label_a_dir, name))
        label_b = pil_load(os.path.join(self.label_b_dir, name))

        img_a, img_b, label_a, label_b = self.transform(img_a, img_b, label_a, label_b)

        return img_a, img_b, label_a, label_b, name