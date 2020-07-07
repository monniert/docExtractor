import cv2
from PIL import Image, ImageDraw, ImageFilter
from pdf2image import convert_from_path

import numpy as np

from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir, get_files_from_dir
from utils.constant import (LABEL_TO_COLOR_MAPPING, COLOR_TO_LABEL_MAPPING)
from utils.logger import print_info


def resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS):
    if isinstance(size, (int, float)):
        assert keep_aspect_ratio
        ratio = float(np.sqrt(size / (img.size[0] * img.size[1])))
        size = round(ratio * img.size[0]), round(ratio * img.size[1])
    elif keep_aspect_ratio:
        ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        size = round(ratio * img.size[0]), round(ratio * img.size[1])

    return img.resize(size, resample=resample)


def draw_line(image, position, color=(0, 0, 0), width=3, blur_radius=2, std_gaussian_noise=(10, 10, 10)):
    canvas = image.copy()
    mask = Image.new(mode='1', size=image.size)
    draw = ImageDraw.Draw(mask)
    draw.line(position, fill=1, width=width)
    canvas.paste(Image.fromarray(cv2.randn(np.array(canvas), mean=color, stddev=std_gaussian_noise)), mask=mask)
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))

    draw = ImageDraw.Draw(mask)
    draw.line(position, fill=1, width=width)
    image.paste(canvas, mask=mask)


def paste_with_blured_borders(canvas, img, position=(0, 0), border_width=3):
    canvas.paste(img, position)
    mask = Image.new('L', canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    x0, y0 = [position[k] - border_width for k in range(2)]
    x1, y1 = [position[k] + img.size[k] + border_width for k in range(2)]

    diam = 2 * border_width
    for d in range(diam + border_width):
        x1, y1 = x1 - 1, y1 - 1
        alpha = 255 if d < border_width else int(255 * (diam + border_width - d) / diam)
        fill = None if d != diam + border_width - 1 else 0
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=alpha)
        x0, y0 = x0 + 1, y0 + 1

    blur = canvas.filter(ImageFilter.GaussianBlur(border_width / 2))
    canvas.paste(blur, mask=mask)


class Image2LabeledArray:
    """Convert png files to 2D labeled array given a color_label_mapping."""

    def __init__(self, input_dir, output_dir, color_label_mapping=COLOR_TO_LABEL_MAPPING, img_extension='png',
                 verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.files = get_files_from_dir(self.input_dir, valid_extensions=img_extension)
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.color_label_mapping = color_label_mapping
        self.verbose = verbose

    def run(self):
        for filename in self.files:
            if self.verbose:
                print_info('Converting and saving as segmentation map {}'.format(filename))
            img = self.convert(Image.open(filename), self.color_label_mapping)
            np.save(self.output_dir / filename.stem, img)

    @staticmethod
    def convert(img, color_label_mapping=COLOR_TO_LABEL_MAPPING):
        arr = np.array(img)
        res = np.zeros(arr.shape[:2], dtype=np.uint8)
        for color, label in color_label_mapping.items():
            res[(arr == color).all(axis=-1)] = label

        return res


class LabeledArray2Image:
    """Convert 2D labeled array to an image given a label_color_mapping."""

    def __init__(self, input_dir, output_dir, label_color_mapping=LABEL_TO_COLOR_MAPPING,
                 img_extension='png', verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.files = get_files_from_dir(self.input_dir, valid_extensions='npy')
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.label_color_mapping = label_color_mapping
        self.extension = img_extension
        self.verbose = verbose

    def run(self):
        for filename in self.files:
            if self.verbose:
                print_info('Converting and saving as segmentation map {}'.format(filename))
            img = self.convert(np.load(filename), self.label_color_mapping)
            img.save(self.output_dir / '{}.{}'.format(filename.stem, self.extension))

    @staticmethod
    def convert(arr, label_color_mapping):
        res = np.zeros(arr.shape + (3,), dtype=np.uint8)
        for label, color in label_color_mapping.items():
            res[arr == label] = color

        return Image.fromarray(res)


class Pdf2Image:
    """
    Convert pdf files in a given input_dir to images. For each pdf, a new eponymous folder would be created and would
    contained one image per pdf page.
    """

    def __init__(self, input_dir, output_dir, suffix_fmt='-{}', out_ext='jpg', create_sub_dir=False, verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.files = get_files_from_dir(self.input_dir, valid_extensions='pdf')
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.suffix_fmt = suffix_fmt
        self.out_ext = out_ext
        self.create_sub_dir = create_sub_dir
        self.verbose = verbose
        if self.verbose:
            print_info("Pdf2Image initialised: found {} files".format(len(self.files)))

    def run(self):
        for filename in self.files:
            if self.verbose:
                print_info('Processing {}'.format(filename.name))
            pages = self.convert(filename)
            max_page_id = len(str(len(pages)))
            path = self.output_dir
            if self.create_sub_dir:
                path = path / str(filename.stem)
                path.mkdir()
            for k, page in enumerate(pages):
                suffix = self.suffix_fmt.format(str(k + 1).zfill(max_page_id))
                page.save(path / '{}{}.{}'.format(filename.stem, suffix, self.out_ext))

    @staticmethod
    def convert(filename, dpi=100):
        filename = coerce_to_path_and_check_exist(filename)
        return convert_from_path(filename, dpi=dpi, use_cropbox=True, fmt='jpg')
