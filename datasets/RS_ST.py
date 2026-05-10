import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import random
import PIL
import math
import collections
from PIL import Image
import sys
import numbers

num_classes = 7
# root = r'/home/sht/Datasets/LandsatSCD512/'
root = r'/home/sht/Datasets/SECONDbi/'

# SECOND
ST_COLORMAP = [[255,255,255], [0,128,0], [128,128,128], [0,255,0], [0,0,255], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'low vegetation', 'ground', 'tree', 'water', 'building', 'sports field']
# LandsatSCD
# ST_COLORMAP = [[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]]
# ST_CLASSES = ['unchanged', 'farmland', 'desert', 'building', 'water']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])



colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def tensor2int(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = im * STD_A + MEAN_A
    else:
        im = im * STD_B + MEAN_B
    return im.astype(np.uint8)

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

def read_RSimages(mode):
    #assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    #label_A_dir = os.path.join(root, mode, 'label1_rgb')
    #label_B_dir = os.path.join(root, mode, 'label2_rgb')
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for idx, it in enumerate(data_list):
        # print(it)
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            
            #print(img_B_path)
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)            
            labels_A.append(label_A)
            labels_B.append(label_B)
        if not idx%500: print('%d/%d images loaded.'%(idx, len(data_list)))
        #if idx>50: break
    
    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, labels_A, labels_B

class Data(data.Dataset):
    def __init__(self, mode, random_flip=False, random_swap=False):
        self.mode = mode
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)
    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_id = os.path.basename(self.imgs_list_A[idx])
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        if self.mode=='train' and self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD(img_A, img_B, label_A, label_B)
        if self.mode=='train' and self.random_swap:
            img_A, img_B, label_A, label_B = transform.rand_swap_SCD(img_A, img_B, label_A, label_B)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B), img_id

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len

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
            return F.to_tensor(imga),\
                    F.to_tensor(imgb),\
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
        return F.normalize(imga, self.mean, self.std, inplace=True),\
               F.normalize(imgb, self.mean, self.std, inplace=True),\
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
        return F.resized_crop(imga, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(imgb, i, j, h, w, self.size, self.interpolation, antialias=True),\
               F.resized_crop(lbla, i, j, h, w, self.size, InterpolationMode.NEAREST),\
               F.resized_crop(lblb, i, j, h, w, self.size, InterpolationMode.NEAREST)


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
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

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

def is_image_file(img_ext):
    return img_ext == 'tif' or img_ext == 'tiff' or img_ext == 'png' or img_ext == 'jpg' or img_ext == 'jpeg' or img_ext == 'JPEG'

def pilload(img_path):
    if is_image_file(img_path.split('.')[-1]):
        img = Image.open(img_path)
        return img

class DataPerAAUG(data.Dataset):
    def __init__(self, mode, random_flip=False, random_swap=False, path=None):
        self.mode = mode
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.img_A_dir = os.path.join(path, mode, 'im1')
        self.img_B_dir = os.path.join(path, mode, 'im2')
        self.label_A_dir = os.path.join(path, mode, 'label1')
        self.label_B_dir = os.path.join(path, mode, 'label2')
        self.imgs_name_list = os.listdir(self.img_A_dir)
        # PerA
        # self.mean = [0.3585, 0.3741, 0.3155]
        # self.std = [0.1483, 0.1283, 0.1198]
        # satmae
        # self.mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        # self.std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        # vmamba
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if mode =='train':
            self.trans = CDMCompose([
                                CDMRandomFlipRotate(),
                                # CDMRandomResizedCrop(size=512, 
                                #                     scale=(0.4, 1.0), 
                                #                     ratio=(0.75, 1.33),),
                                CDMColorJitter(brightness=0.2, 
                                                contrast=0.2, 
                                                saturation=0.1, 
                                                hue=0.1),
                                CDMToTensor(),
                                CDMNormalize(mean=self.mean, std=self.std)
                               ])
        else:
            self.trans = CDMCompose([CDMToTensor(),
                                CDMNormalize(mean=self.mean, std=self.std)])
    
    # def get_mask_name(self, idx):
    #     mask_name = os.path.split(self.imgs_name_list[idx])[-1]
    #     return mask_name

    def __getitem__(self, idx):
        img_id = os.path.basename(self.imgs_name_list[idx])
        img_A = pilload(os.path.join(self.img_A_dir, self.imgs_name_list[idx]))
        img_B = pilload(os.path.join(self.img_B_dir, self.imgs_name_list[idx]))
        label_A = pilload(os.path.join(self.label_A_dir, self.imgs_name_list[idx]))
        label_B = pilload(os.path.join(self.label_B_dir, self.imgs_name_list[idx]))

        auged_img_A, auged_img_B, auged_label_A, auged_label_B = self.trans(img_A, img_B, label_A, label_B)
        return auged_img_A, auged_img_B, auged_label_A, auged_label_B, img_id

    def __len__(self):
        return len(self.imgs_name_list)
    

class DataPerA(data.Dataset):
    def __init__(self, mode, random_flip=False, random_swap=False):
        self.mode = mode
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)
        self.trans = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.3585, 0.3741, 0.3155], std=[0.1483, 0.1283, 0.1198])
            ])
    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_id = os.path.basename(self.imgs_list_A[idx])
        img_A = io.imread(self.imgs_list_A[idx])
        img_B = io.imread(self.imgs_list_B[idx])
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        if self.mode=='train' and self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_SCD(img_A, img_B, label_A, label_B)
        if self.mode=='train' and self.random_swap:
            img_A, img_B, label_A, label_B = transform.rand_swap_SCD(img_A, img_B, label_A, label_B)
        return self.trans(img_A), self.trans(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B), img_id

    def __len__(self):
        return len(self.imgs_list_A)



class CDM4Normalize(object):
    """
    分别对 A 和 B 使用不同的 mean 和 std
    """
    def __init__(self, mean_A, std_A, mean_B, std_B):
        self.mean_A = mean_A
        self.std_A = std_A
        self.mean_B = mean_B
        self.std_B = std_B

    def __call__(self, imga, imgb, lbla, lblb):
        imga = F.normalize(imga, self.mean_A, self.std_A, inplace=True)
        imgb = F.normalize(imgb, self.mean_B, self.std_B, inplace=True)
        return imga, imgb, lbla, lblb

class DataAUG(data.Dataset):
    def __init__(self, mode, random_flip=False, random_swap=False, path=None):
        self.mode = mode
        self.random_flip = random_flip
        self.random_swap = random_swap
        self.img_A_dir = os.path.join(path, mode, 'im1')
        self.img_B_dir = os.path.join(path, mode, 'im2')
        self.label_A_dir = os.path.join(path, mode, 'label1')
        self.label_B_dir = os.path.join(path, mode, 'label2')
        self.imgs_name_list = os.listdir(self.img_A_dir)
        self.mean_A = (MEAN_A / 255.).tolist()
        self.std_A  = (STD_A / 255.).tolist()

        self.mean_B = (MEAN_B / 255.).tolist()
        self.std_B  = (STD_B / 255.).tolist()
        if mode == 'train':
            self.trans = CDMCompose([
                CDMRandomFlipRotate(),
                CDMColorJitter(brightness=0.2, 
                            contrast=0.2, 
                            saturation=0.1, 
                            hue=0.1),
                CDMToTensor(),
                CDM4Normalize(mean_A=self.mean_A, std_A=self.std_A,
                            mean_B=self.mean_B, std_B=self.std_B)
            ])
        else:
            self.trans = CDMCompose([
                CDMToTensor(),
                CDM4Normalize(mean_A=self.mean_A, std_A=self.std_A,
                            mean_B=self.mean_B, std_B=self.std_B)
            ])
    
    # def get_mask_name(self, idx):
    #     mask_name = os.path.split(self.imgs_name_list[idx])[-1]
    #     return mask_name

    def __getitem__(self, idx):
        img_id = os.path.basename(self.imgs_name_list[idx])
        img_A = pilload(os.path.join(self.img_A_dir, self.imgs_name_list[idx]))
        img_B = pilload(os.path.join(self.img_B_dir, self.imgs_name_list[idx]))
        label_A = pilload(os.path.join(self.label_A_dir, self.imgs_name_list[idx]))
        label_B = pilload(os.path.join(self.label_B_dir, self.imgs_name_list[idx]))

        auged_img_A, auged_img_B, auged_label_A, auged_label_B = self.trans(img_A, img_B, label_A, label_B)
        return auged_img_A, auged_img_B, auged_label_A, auged_label_B, img_id

    def __len__(self):
        return len(self.imgs_name_list)