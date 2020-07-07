from abc import ABCMeta, abstractproperty
import cv2
from PIL import Image, ImageEnhance, ImageFilter

import numpy as np
from numpy.random import choice, uniform, randint
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from skimage.morphology import skeletonize

from utils import coerce_to_path_and_check_exist, get_files_from_dir
from utils.constant import (BACKGROUND_LABEL, BASELINE_COLOR, CONTEXT_BACKGROUND_COLOR, CONTEXT_BACKGROUND_LABEL,
                            LABEL_TO_COLOR_MAPPING, SEG_GROUND_TRUTH_FMT, TEXT_BORDER_COLOR)
from utils.image import resize
from utils.path import DATASETS_PATH

INPUT_EXTENSIONS = ['jpeg', 'jpg', 'JPG']
LABEL_EXTENSION = 'png'

# Data augmentations
BLUR_RADIUS_RANGE = (0, 0.5)
BRIGHTNESS_FACTOR_RANGE = (0.9, 1.1)
CONTRAST_FACTOR_RANGE = (0.5, 1.5)
ROTATION_ANGLE_RANGE = (-10, 10)
SAMPLING_RATIO_RANGE = (0.6, 1.4)
TRANPOSITION_CHOICES = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
TRANPOSITION_WEIGHTS = [0.25, 0.25, 0.25, 0.25]


class AbstractSegDataset(TorchDataset):
    """Abstract torch dataset for segmentation task."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def name(self):
        pass

    @property
    def root_path(self):
        return DATASETS_PATH / self.name

    def __init__(self, split, restricted_labels, **kwargs):
        self.data_path = coerce_to_path_and_check_exist(self.root_path) / split
        self.split = split
        self.input_files, self.label_files = self._get_input_label_files()
        self.size = len(self.input_files)
        self.restricted_labels = sorted(restricted_labels)
        self.restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in self.restricted_labels]
        self.label_idx_color_mapping = {self.restricted_labels.index(l) + 1: c
                                        for l, c in zip(self.restricted_labels, self.restricted_colors)}
        self.color_label_idx_mapping = {c: l for l, c in self.label_idx_color_mapping.items()}
        self.fill_background = BACKGROUND_LABEL in self.restricted_labels
        self.n_classes = len(self.restricted_labels) + 1
        self.img_size = kwargs.get('img_size')
        self.keep_aspect_ratio = kwargs.get('keep_aspect_ratio', True)
        self.baseline_dilation_iter = kwargs.get('baseline_dilation_iter', 1)
        self.normalize = kwargs.get('normalize', True)
        self.data_augmentation = kwargs.get('data_augmentation', True) and split == 'train'
        self.blur_radius_range = kwargs.get('blur_radius_range', BLUR_RADIUS_RANGE)
        self.brightness_factor_range = kwargs.get('brightness_factor_range', BRIGHTNESS_FACTOR_RANGE)
        self.contrast_factor_range = kwargs.get('contrast_factor_range', CONTRAST_FACTOR_RANGE)
        self.rotation_angle_range = kwargs.get('rotation_angle_range', ROTATION_ANGLE_RANGE)
        self.sampling_ratio_range = kwargs.get('sampling_ratio_range', SAMPLING_RATIO_RANGE)
        self.sampling_max_nb_pixels = kwargs.get('sampling_max_nb_pixels')
        self.transposition_weights = kwargs.get('transposition_weights', TRANPOSITION_WEIGHTS)

    def _get_input_label_files(self):
        input_files = get_files_from_dir(self.data_path, INPUT_EXTENSIONS, sort=True)
        label_files = get_files_from_dir(self.data_path, [LABEL_EXTENSION])

        if len(label_files) == 0 and self.split == 'test':
            return input_files, None
        elif len(input_files) != len(label_files):
            raise RuntimeError("The number of inputs and labels don't match")

        if len(input_files) < 1e5:
            inputs = [p.stem for p in input_files]
            labels = [str(p.name) for p in label_files]
            invalid = []
            for name in inputs:
                if SEG_GROUND_TRUTH_FMT.format(name, LABEL_EXTENSION) not in labels:
                    invalid.append(name)
            if len(invalid) > 0:
                raise FileNotFoundError("Some inputs don't have corresponding labels: {}".format(' '.join(invalid)))
        else:
            assert len(input_files) == len(label_files)

        label_files = [path.parent / SEG_GROUND_TRUTH_FMT.format(path.stem, LABEL_EXTENSION) for path in input_files]
        return input_files, label_files

    @property
    def metric_labels(self):
        return [CONTEXT_BACKGROUND_LABEL] + self.restricted_labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.data_augmentation:
            augm_kwargs = {
                'blur_radius': uniform(*self.blur_radius_range),
                'brightness': uniform(*self.brightness_factor_range),
                'contrast': uniform(*self.contrast_factor_range),
                'rotation': randint(*self.rotation_angle_range),
                'sampling_ratio': uniform(*self.sampling_ratio_range),
                'transpose': choice(TRANPOSITION_CHOICES, p=self.transposition_weights),
            }
        else:
            augm_kwargs = {}

        inp = np.array(self.transform(Image.open(self.input_files[idx]), **augm_kwargs), dtype=np.float32) / 255
        if self.normalize:
            inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
        inp = np.dstack([inp, inp, inp]) if len(inp.shape) == 2 else inp
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float()  # HWC -> CHW tensor

        if self.label_files is None:
            label = None
        else:
            img = Image.open(self.label_files[idx])
            arr_segmap = np.array(self.transform(img, is_gt=True, **augm_kwargs), dtype=np.uint8)
            unique_colors = set([color for size, color in img.getcolors()]).difference({CONTEXT_BACKGROUND_COLOR})
            label = self.encode_segmap(arr_segmap, unique_colors)

        return inp, label

    def transform(self, img, is_gt=False, **augm_kwargs):
        if self.img_size is not None:
            resample = Image.NEAREST if is_gt else Image.ANTIALIAS
            if self.data_augmentation:
                size = tuple(map(lambda s: round(augm_kwargs['sampling_ratio'] * s), self.img_size))
                if self.sampling_max_nb_pixels is not None and self.keep_aspect_ratio:
                    ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))
                    real_size = round(ratio * img.size[0]), round(ratio * img.size[1])
                    nb_pixels = np.product(real_size)
                    if nb_pixels > self.sampling_max_nb_pixels:
                        ratio = float(np.sqrt(self.sampling_max_nb_pixels / nb_pixels))
                        size = round(ratio * real_size[0]), round(ratio * real_size[1])
            else:
                size = self.img_size
            img = resize(img, size=size, keep_aspect_ratio=self.keep_aspect_ratio, resample=resample)

        if self.data_augmentation:
            if not is_gt:
                img = img.filter(ImageFilter.GaussianBlur(radius=augm_kwargs['blur_radius']))
                img = ImageEnhance.Brightness(img).enhance(augm_kwargs['brightness'])
                img = ImageEnhance.Contrast(img).enhance(augm_kwargs['contrast'])

            resample = Image.NEAREST if is_gt else Image.BICUBIC
            img = img.rotate(augm_kwargs['rotation'], resample=resample, fillcolor=CONTEXT_BACKGROUND_COLOR)
            if augm_kwargs['transpose'] is not None:
                img = img.transpose(augm_kwargs['transpose'])

        return img

    def encode_segmap(self, arr_segmap, unique_colors=None):
        if unique_colors is None:
            unique_colors = set(map(tuple, list(np.unique(arr_segmap.reshape(-1, arr_segmap.shape[2]), axis=0)))) \
                .difference({CONTEXT_BACKGROUND_COLOR})

        label = np.zeros(arr_segmap.shape[:2], dtype=np.uint8)
        for color in unique_colors:
            if color in self.restricted_colors or self.fill_background:
                if color == TEXT_BORDER_COLOR and BASELINE_COLOR in self.restricted_colors and \
                        TEXT_BORDER_COLOR in self.restricted_colors:
                    continue
                mask = (arr_segmap == color).all(axis=-1)
                if color == BASELINE_COLOR:
                    sklt = skeletonize(mask).astype(np.uint8)
                    kernel = np.ones((3, 3))
                    d_iter = self.baseline_dilation_iter
                    mask = cv2.dilate(sklt, kernel, iterations=d_iter).astype(np.bool)
                    if TEXT_BORDER_COLOR in self.restricted_colors:
                        border_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=d_iter+1).astype(np.bool)
                        label[border_mask] = self.color_label_idx_mapping[TEXT_BORDER_COLOR]
                label[mask] = self.color_label_idx_mapping.get(color, BACKGROUND_LABEL)
        label = torch.from_numpy(label).long()

        return label
