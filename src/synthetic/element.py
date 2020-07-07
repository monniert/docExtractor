from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter
import string

import cv2
import numpy as np
from numpy.random import uniform, choice
from random import randint, choice as rand_choice

import arabic_reshaper
from bidi.algorithm import get_display
from translation import google
from unidecode import unidecode

from utils import use_seed
from utils.constant import (BACKGROUND_COLOR, BASELINE_COLOR, CAPTION_COLOR, CONTEXT_BACKGROUND_COLOR, DRAWING_COLOR,
                            FLOATING_WORD_COLOR, GLYPH_COLOR, IMAGE_COLOR, PARAGRAPH_COLOR, TABLE_WORD_COLOR,
                            TITLE_COLOR, TEXT_BORDER_COLOR)
from utils.constant import (BACKGROUND_LABEL, BASELINE_LABEL, CAPTION_LABEL, CONTEXT_BACKGROUND_LABEL, DRAWING_LABEL,
                            FLOATING_WORD_LABEL, GLYPH_LABEL, IMAGE_LABEL, PARAGRAPH_LABEL, TABLE_WORD_LABEL,
                            TITLE_LABEL, TEXT_BORDER_LABEL)
from utils.constant import SEG_GROUND_TRUTH_FMT
from utils.image import paste_with_blured_borders, resize
from utils.path import SYNTHETIC_RESRC_PATH
from synthetic.resource import (ResourceDatabase, BACKGROUND_RESRC_NAME, CONTEXT_BACKGROUND_RESRC_NAME,
                                DRAWING_RESRC_NAME, DRAWING_BACKGROUND_RESRC_NAME, GLYPH_FONT_RESRC_NAME,
                                FONT_RESRC_NAME, IMAGE_RESRC_NAME, NOISE_PATTERN_RESRC_NAME, TEXT_RESRC_NAME)


DATABASE = ResourceDatabase()

BLURED_BORDER_WIDTH_RANGE = (1, 7)
GAUSSIAN_NOISE_STD_RANGE = (2, 10)
NOISE_PATTERN_SIZE_RANGE = {
    'border_hole': (100, 600),
    'center_hole': (100, 400),
    'corner_hole': (100, 400),
    'phantom_character': (30, 100),
}
NOISE_PATTERN_OPACITY_RANGE = (0.2, 0.6)
POS_ELEMENT_OPACITY_RANGE = {
    'drawing': (200, 255),
    'glyph': (150, 255),
    'image': (150, 255),
    'table': (200, 255),
    'text': (200, 255),
}
NEG_ELEMENT_OPACITY_RANGE = {
    'drawing': (0, 10),
    'glyph': (0, 10),
    'image': (0, 25),
    'table': (0, 10),
    'text': (0, 10),
}
NEG_ELEMENT_BLUR_RADIUS_RANGE = (1, 2.5)

BACKGROUND_BLUR_RADIUS_RANGE = (0, 0.2)
BACKGROUND_COLOR_BLEND_FREQ = 0.1
CONTEXT_BACKGROUND_UNIFORM_FREQ = 0.5
DRAWING_CONTRAST_FACTOR_RANGE = (1, 4)
DRAWING_WITH_BACKGROUND_FREQ = 0.3
DRAWING_WITH_COLOR_FREQ = 0.3
GLYPH_COLORED_FREQ = 0.5
LINE_WIDTH_RANGE = (1, 4)
TABLE_LAYOUT_RANGE = {
    'col_size_range': (50, 200),
}

TEXT_BASELINE_HEIGHT = 5
TEXT_BBOX_FREQ = 0.3
TEXT_BBOX_BORDER_WIDTH_RANGE = (1, 6)
TEXT_BBOX_PADDING_RANGE = (0, 20)
TEXT_COLORED_FREQ = 0.5
TEXT_FONT_TYPE_RATIO = {
    'arabic': 0.1,
    'chinese': 0.1,
    'handwritten': 0.4,
    'normal': 0.4,
}
TEXT_JUSTIFIED_PARAGRAPH_FREQ = 0.7
TEXT_ROTATION_ANGLE_RANGE = (-60, 60)
TEXT_TIGHT_PARAGRAPH_FREQ = 0.5
TEXT_TITLE_UPPERCASE_RATIO = 0.5
TEXT_TITLE_UNILINE_RATIO = 0.25
TEXT_UNDERLINED_FREQ = 0.1
TEXT_UNDERLINED_PADDING_RANGE = (0, 4)


@use_seed()
def get_random_noise_pattern(width, height):
    pattern_path = choice(DATABASE[NOISE_PATTERN_RESRC_NAME])
    pattern_type = Path(pattern_path).parent.name
    img = Image.open(pattern_path).convert('L')
    size_min, size_max = NOISE_PATTERN_SIZE_RANGE[pattern_type]
    size_max = min(min(width, height), size_max)
    size = (randint(size_min, size_max), randint(size_min, size_max))
    if pattern_type in ['border_hole', 'corner_hole']:
        img = resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS)
        rotation = choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rotation is not None:
            img = img.transpose(rotation)
        if pattern_type == 'border_hole':
            if rotation is None:
                position = ((randint(0, width - img.size[0]), 0))
            elif rotation == Image.ROTATE_90:
                position = (0, randint(0, height - img.size[1]))
            elif rotation == Image.ROTATE_180:
                position = ((randint(0, width - img.size[0]), height - img.size[1]))
            else:
                position = (width - img.size[0], randint(0, height - img.size[1]))
        else:
            if rotation is None:
                position = (0, 0)
            elif rotation == Image.ROTATE_90:
                position = (0, height - img.size[1])
            elif rotation == Image.ROTATE_180:
                position = (width - img.size[0], height - img.size[1])
            else:
                position = (width - img.size[0], 0)
    else:
        img = resize(img, size, keep_aspect_ratio=False, resample=Image.ANTIALIAS)
        rotation = randint(0, 360)
        img = img.rotate(rotation, fillcolor=255)
        pad = max(img.width, img.height)
        position = (randint(0, max(0, width - pad)), randint(0, max(0, height - pad)))

    alpha = uniform(*NOISE_PATTERN_OPACITY_RANGE)
    arr = np.array(img.convert('RGBA'))
    arr[:, :, 3] = (255 - arr[:, :, 2]) * alpha
    hue_color = randint(0, 360)
    value_ratio = uniform(0.95, 1)
    return Image.fromarray(arr), hue_color, value_ratio, position


class AbstractElement:
    """Abstract class that defines the characteristics of a document's element."""
    __metaclass__ = ABCMeta

    label = NotImplemented
    color = NotImplemented
    content_width = NotImplemented
    content_height = NotImplemented
    name = NotImplemented
    pos_x = NotImplemented
    pos_y = NotImplemented

    def __init__(self, width, height, seed=None, **kwargs):
        self.width, self.height = width, height
        self.parameters = kwargs
        self.generate_content(seed=seed)

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def content_size(self):
        return (self.content_width, self.content_height)

    @property
    def position(self):
        return (self.pos_x, self.pos_y)

    @use_seed()
    @abstractmethod
    def generate_content(self):
        pass

    @abstractmethod
    def to_image(self):
        pass

    def to_image_as_array(self):
        return np.array(self.to_image(), dtype=np.float32) / 255

    @abstractmethod
    def to_label_as_array(self):
        pass

    def to_label_as_img(self):
        arr = self.to_label_as_array()
        res = np.zeros(arr.shape + (3,), dtype=np.uint8)
        res[arr == self.label] = self.color
        return Image.fromarray(res)


class BackgroundElement(AbstractElement):
    label = BACKGROUND_LABEL
    color = BACKGROUND_COLOR
    name = 'background'

    @use_seed()
    def generate_content(self):
        self.img_path = self.parameters.get('image_path') or choice(DATABASE[BACKGROUND_RESRC_NAME])
        self.img = Image.open(self.img_path).resize(self.size, resample=Image.ANTIALIAS).convert('RGB')
        self.blur_radius = uniform(*BACKGROUND_BLUR_RADIUS_RANGE)
        self.content_width, self.content_height = self.size
        self.pos_x, self.pos_y = (0, 0)

        color_blend = choice([True, False], p=[BACKGROUND_COLOR_BLEND_FREQ, 1 - BACKGROUND_COLOR_BLEND_FREQ])
        if color_blend:
            new_img = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2HSV)
            new_img[:, :, 0] = randint(0, 360)
            self.img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB))

    def to_image(self, flip=False):
        if flip:
            return self.img.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(self.blur_radius))
        else:
            return self.img.filter(ImageFilter.GaussianBlur(self.blur_radius))

    def to_label_as_array(self):
        return np.full(self.size, self.label, dtype=np.uint8).transpose()

    @property
    def inherent_left_margin(self):
        img_path = Path(self.img_path) if isinstance(self.img_path, str) else self.img_path
        try:
            return int(int(img_path.parent.name) * self.width / 596)  # XXX: margins were calibrated on 596x842 images
        except ValueError:
            return 0


class ContextBackgroundElement(AbstractElement):
    label = CONTEXT_BACKGROUND_LABEL
    color = CONTEXT_BACKGROUND_COLOR
    name = 'context_background'

    @use_seed()
    def generate_content(self):
        uniform_bg = choice([True, False], p=[CONTEXT_BACKGROUND_UNIFORM_FREQ, 1 - CONTEXT_BACKGROUND_UNIFORM_FREQ])
        if uniform_bg:
            color = randint(0, 255)
            std = randint(*GAUSSIAN_NOISE_STD_RANGE)
            img = Image.new(mode='L', color=color, size=self.size)
            img = Image.fromarray(cv2.randn(np.array(img), mean=color, stddev=std))
        else:
            color = None
            img_path = self.parameters.get('image_path') or choice(DATABASE[CONTEXT_BACKGROUND_RESRC_NAME])
            img = Image.open(img_path)

        transpose_idx = choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if transpose_idx is not None:
            img = img.transpose(transpose_idx)

        self.intensity = img.convert('L').resize((1, 1)).getpixel((0, 0))
        self.img = img.resize(self.size, resample=Image.ANTIALIAS).convert('RGB')
        self.blur_radius = uniform(*BACKGROUND_BLUR_RADIUS_RANGE)
        self.content_width, self.content_height = self.size
        self.pos_x, self.pos_y = (0, 0)

    def to_image(self):
        return self.img.filter(ImageFilter.GaussianBlur(self.blur_radius))

    def to_label_as_array(self):
        return np.full(self.size, self.label, dtype=np.uint8).transpose()


class DrawingElement(AbstractElement):
    label = DRAWING_LABEL
    color = DRAWING_COLOR
    name = 'drawing'

    @use_seed()
    def generate_content(self):
        self.img_path = self.parameters.get('image_path') or choice(DATABASE[DRAWING_RESRC_NAME])
        img = Image.open(self.img_path).convert('L')
        self.contrast_factor = uniform(*DRAWING_CONTRAST_FACTOR_RANGE)
        self.as_negative = self.parameters.get('as_negative', False)
        self.blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        self.opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name] if self.as_negative
                               else POS_ELEMENT_OPACITY_RANGE[self.name])
        self.colored = choice([True, False], p=[DRAWING_WITH_COLOR_FREQ, 1 - DRAWING_WITH_COLOR_FREQ])
        if self.colored:
            self.color_channels = choice(range(3), randint(1, 2), replace=False)
            self.other_channel_intensity = [randint(0, 100) for _ in range(3)]
            self.hue_color = randint(0, 360)
        else:
            self.color_channels, self.color_intensity = None, None

        self.with_background = choice([True, False], p=[DRAWING_WITH_BACKGROUND_FREQ, 1 - DRAWING_WITH_BACKGROUND_FREQ])
        if self.with_background:
            self.color, self.label = IMAGE_COLOR, IMAGE_LABEL
            blured_border_width = randint(*BLURED_BORDER_WIDTH_RANGE)
            max_size = [s - 2 * blured_border_width for s in self.size]
            img = resize(img, max_size)
            bg = Image.open(choice(DATABASE[DRAWING_BACKGROUND_RESRC_NAME])).resize(img.size)
            new_img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2HSV)
            new_img[:, :, 0] = randint(0, 360)
            background = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB))
            if not self.colored:
                background = background.convert('L').convert('RGB')
            self.background = background
            self.blured_border_width = blured_border_width
        else:
            img = resize(img, self.size)
            self.background, self.blured_border_width = None, 0

        self.img = img
        self.content_width, self.content_height = self.img.size
        self.pos_x = randint(0, self.width - self.content_width)
        self.pos_y = randint(0, self.height - self.content_height)

        label_path = Path(self.img_path).parent / SEG_GROUND_TRUTH_FMT.format(Path(self.img_path).stem, 'png')
        self.mask_label = np.array(resize(Image.open(label_path), self.img.size, False, resample=Image.NEAREST))

    def scaled_size(self, img):
        size = [s - 2 * self.blured_border_width for s in self.size]
        ratio = img.size[0] / img.size[1]
        return map(min, zip(*[size, (int(size[1] * ratio), int(size[0] / ratio))]))

    def to_image(self, canvas=None):
        if canvas is None:
            canvas = Image.new('RGB', self.size, (255, 255, 255))
        if self.with_background:
            paste_with_blured_borders(canvas, self.background, self.position, border_width=self.blured_border_width)

        canvas_arr = np.array(canvas.convert('RGB'))
        enhanced_img = ImageEnhance.Contrast(self.img).enhance(self.contrast_factor)
        img_arr = np.array(enhanced_img, dtype=np.uint8)
        img_arr[self.mask_label == 0] = 255
        if self.colored:
            img_arr_channels = []
            for i in range(3):
                if i in self.color_channels:
                    img_arr_channels.append(img_arr)
                else:
                    other_arr = img_arr.copy()
                    other_arr[img_arr != 255] = self.other_channel_intensity[i]
                    img_arr_channels.append(other_arr)
            img_arr_channels_hsv = cv2.cvtColor(np.dstack(img_arr_channels), cv2.COLOR_RGB2HSV)
            img_arr_channels_hsv[:, :, 0] = self.hue_color
            img_arr_channels = cv2.cvtColor(img_arr_channels_hsv, cv2.COLOR_HSV2RGB)
        else:
            img_arr_channels = np.dstack([img_arr for i in range(3)])

        x, y = self.position
        img_arr_rgb = np.full(canvas_arr.shape, 255, dtype=np.uint8)
        img_arr_rgb[y:y+self.content_height, x:x+self.content_width] = img_arr_channels
        result = Image.fromarray(cv2.multiply(canvas_arr, img_arr_rgb, scale=1/255)).convert('RGBA')
        result.putalpha(self.opacity)

        if self.as_negative:
            result = result.filter(ImageFilter.GaussianBlur(self.blur_radius))
        return result

    def to_label_as_array(self):
        label = np.full(self.size, BACKGROUND_LABEL, dtype=np.uint8)
        if self.as_negative:
            return label.transpose()
        else:
            x, y = self.position
            if self.with_background:
                label[x:x+self.content_width, y:y+self.content_height] = self.label
            else:
                mask = (self.mask_label == 255).transpose()
                label[x:x+self.content_width, y:y+self.content_height][mask] = self.label
            return label.transpose()


class GlyphElement(AbstractElement):
    label = GLYPH_LABEL
    color = GLYPH_COLOR
    font_size_range = (200, 800)
    name = 'glyph'

    @use_seed()
    def generate_content(self):
        self.font_path = choice(DATABASE[GLYPH_FONT_RESRC_NAME])
        self.letter = self.parameters.get('letter') or rand_choice(string.ascii_uppercase)

        # To avoid oversized letters
        rescaled_height = (self.height * 2) // 3
        min_fs, max_fs = self.font_size_range
        actual_max_fs = min(rescaled_height, max_fs)
        tmp_font = ImageFont.truetype(self.font_path, size=actual_max_fs)
        while tmp_font.getsize(self.letter)[0] > self.width and actual_max_fs > self.font_size_range[0]:
            actual_max_fs -= 1
            tmp_font = ImageFont.truetype(self.font_path, size=actual_max_fs)
        if min_fs < actual_max_fs:
            self.font_size = randint(min_fs, actual_max_fs)
        else:
            self.font_size = actual_max_fs

        self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        self.as_negative = self.parameters.get('as_negative', False)
        self.blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        self.opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name] if self.as_negative
                               else POS_ELEMENT_OPACITY_RANGE[self.name])
        self.colored = choice([True, False], p=[GLYPH_COLORED_FREQ, 1 - GLYPH_COLORED_FREQ])
        self.colors = (0, 0, 0) if not self.colored else tuple([randint(0, 150) for _ in range(3)])
        self.content_width, self.content_height = self.font.getsize(self.letter)
        self.pos_x = randint(0, max(0, self.width - self.content_width))
        self.pos_y = randint(0, max(0, self.height - self.content_height))

    def to_image(self):
        canvas = Image.new('RGBA', self.size)
        image_draw = ImageDraw.Draw(canvas)
        colors_alpha = self.colors + (self.opacity,)
        image_draw.text(self.position, self.letter, font=self.font, fill=colors_alpha)
        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))
        return canvas

    def to_label_as_array(self):
        if self.as_negative:
            return np.full(self.size, BACKGROUND_LABEL, dtype=np.uint8).transpose()
        else:
            padding = self.font_size  # XXX we want to avoid borders when computing closings
            size = tuple(map(lambda s: s + 2 * padding, self.size))
            position = tuple(map(lambda s: s + padding, self.position))
            canvas = Image.new('L', size, color=0)
            image_draw = ImageDraw.Draw(canvas)
            image_draw.text(position, self.letter, font=self.font, fill=255)

            nb_iter = self.font_size // 2
            label = (np.asarray(canvas, dtype=np.uint8) > 0).astype(np.uint8)
            label = cv2.morphologyEx(label, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=nb_iter)
            label = label[padding:-padding, padding:-padding]
            label[label == 0] = BACKGROUND_LABEL
            label[label == 1] = self.label
            return label


class ImageElement(AbstractElement):
    label = IMAGE_LABEL
    color = IMAGE_COLOR
    name = 'image'

    @use_seed()
    def generate_content(self):
        self.blured_border_width = randint(*BLURED_BORDER_WIDTH_RANGE)
        self.as_negative = self.parameters.get('as_negative', False)
        self.blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        self.opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name] if self.as_negative
                               else POS_ELEMENT_OPACITY_RANGE[self.name])
        img = Image.open(self.parameters.get('image_path') or choice(DATABASE[IMAGE_RESRC_NAME]))
        img.putalpha(self.opacity)

        max_size = [s - 2 * self.blured_border_width for s in self.size]
        self.img = resize(img, max_size)
        self.content_width, self.content_height = self.img.size
        self.pos_x = randint(0, self.width - self.content_width)
        self.pos_y = randint(0, self.height - self.content_height)

    def to_image(self, canvas_color=(255, 255, 255)):
        canvas = Image.new('RGBA', self.size, canvas_color + (0,))
        paste_with_blured_borders(canvas, self.img, self.position, border_width=self.blured_border_width)
        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))
        return canvas

    def to_label_as_array(self):
        label = np.full(self.size, BACKGROUND_LABEL, dtype=np.uint8)
        if self.as_negative:
            return label.transpose()
        else:
            x, y = self.position
            label[x:x+self.content_width, y:y+self.content_height] = self.label
            return label.transpose()


class AbstractTextElement(AbstractElement):
    """Abstract class that defines a text element such as titles, captions and paragraphs."""
    __metaclass__ = ABCMeta
    border_label = TEXT_BORDER_LABEL
    border_color = TEXT_BORDER_COLOR
    name = 'text'

    @abstractproperty
    def text_type(self):
        pass

    @abstractproperty
    def n_max_lines(self):
        pass

    @abstractproperty
    def n_min_characters(self):
        pass

    @abstractproperty
    def font_size_range(self):
        pass

    @abstractproperty
    def line_spacing_range(self):
        pass

    @staticmethod
    def get_random_font():
        font_type = choice(list(TEXT_FONT_TYPE_RATIO.keys()), p=list(TEXT_FONT_TYPE_RATIO.values()))
        return choice(DATABASE[FONT_RESRC_NAME][font_type])

    @use_seed()
    def generate_content(self):
        min_fs, max_fs = self.font_size_range
        min_spacing, max_spacing = self.line_spacing_range
        if self.text_type == 'paragraph':
            tight = choice([True, False], p=[TEXT_TIGHT_PARAGRAPH_FREQ, 1 - TEXT_TIGHT_PARAGRAPH_FREQ])
            if tight:
                max_fs, max_spacing = max(min_fs, 30), max(min_spacing, 4)
            else:
                min_fs, min_spacing = min(30, max_fs), min(2, max_fs)
            self.justified = tight and choice([True, False], p=[TEXT_JUSTIFIED_PARAGRAPH_FREQ,
                                                                1 - TEXT_JUSTIFIED_PARAGRAPH_FREQ])
        else:
            self.justified = False
        self.font_path = self.parameters.get('font_path') or self.get_random_font()
        self.font_type = Path(self.font_path).relative_to(SYNTHETIC_RESRC_PATH / FONT_RESRC_NAME).parts[0]

        # To avoid oversized letters
        rescaled_height = (self.height * 2) // 3
        actual_max_fs = min(rescaled_height, max_fs)
        tmp_font = ImageFont.truetype(self.font_path, size=actual_max_fs)
        while tmp_font.getsize('bucolic')[0] > self.width and actual_max_fs > self.font_size_range[0]:
            actual_max_fs -= 1
            tmp_font = ImageFont.truetype(self.font_path, size=actual_max_fs)
        if min_fs < actual_max_fs:
            self.font_size = randint(min_fs, actual_max_fs)
        else:
            self.font_size = actual_max_fs
        self.spacing = randint(min_spacing, max_spacing)

        if 'text' in self.parameters:
            text = self.parameters['text']
        else:
            n_char = 0
            while (n_char <= self.n_min_characters):
                self.text_path = choice(DATABASE[TEXT_RESRC_NAME])
                with open(self.text_path) as f:
                    text = f.read().rstrip('\n')
                n_char = len(text)

        self.baseline_as_label = self.parameters.get('baseline_as_label', False)
        if self.baseline_as_label:
            self.label, self.color = BASELINE_LABEL, BASELINE_COLOR

        self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        self.as_negative = self.parameters.get('as_negative', False)
        self.blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        self.opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name] if self.as_negative
                               else POS_ELEMENT_OPACITY_RANGE[self.name])
        self.transpose = self.parameters.get('transpose', False)

        if self.text_type == 'title':
            self.uppercase = (choice([True, False], p=[TEXT_TITLE_UPPERCASE_RATIO, 1 - TEXT_TITLE_UPPERCASE_RATIO]) or
                              self.font_type == 'chinese')
            self.uniline = choice([True, False], p=[TEXT_TITLE_UNILINE_RATIO, 1 - TEXT_TITLE_UNILINE_RATIO])
            n_spaces = randint(2, 5)
            text = text.replace(' ', ' ' * n_spaces)
        elif self.text_type == 'word':
            self.uppercase = self.font_type == 'chinese'
            self.uniline = True
        else:
            self.uppercase = self.font_type == 'chinese'
            self.uniline = False

        dark_mode = self.parameters.get('dark_mode', True)
        color_range = (0, 75) if dark_mode else (175, 255)
        colored = choice([True, False], p=[TEXT_COLORED_FREQ, 1 - TEXT_COLORED_FREQ])
        colors = tuple([randint(*color_range)] * 3) if not colored else tuple([randint(*color_range) for _ in range(3)])
        self.colors_alpha = colors + (self.opacity,)

        self.underlined = (choice([True, False], p=[TEXT_UNDERLINED_FREQ, 1 - TEXT_UNDERLINED_FREQ]) and
                           self.font_type in ['normal', 'handwritten'] and not self.text_type == 'word')
        if self.underlined:
            self.underline_params = {
                'width': randint(*LINE_WIDTH_RANGE),
                'fill': tuple([randint(*color_range)] * 3) + (self.opacity,),
            }
            strikethrough = choice([True, False])
            line_height = self.font.font.getsize('a')[0][1]
            self.underline_padding = randint(*TEXT_UNDERLINED_PADDING_RANGE) if not strikethrough else -line_height // 2
        else:
            self.underlined_params, self.underline_padding = None, 0
        self.with_bbox = self.text_type == 'paragraph' and choice([True, False], p=[TEXT_BBOX_FREQ, 1 - TEXT_BBOX_FREQ])
        if self.with_bbox:
            filled = choice([True, False])
            alpha = randint(0, min(self.opacity, 100))
            self.bbox_params = {
                'width': randint(*TEXT_BBOX_BORDER_WIDTH_RANGE),
                'outline': self.colors_alpha,
                'fill': tuple([randint(150, 255) for _ in range(3)]) + (alpha,) if filled else None
            }
            self.padding = randint(*TEXT_BBOX_PADDING_RANGE) + self.bbox_params['width'] + 1
        else:
            self.bbox_params, self.padding = None, 0

        self.with_border_label = self.parameters.get('with_border_label', False)
        if self.with_border_label:
            label_height = self.font.font.getsize('A')[0][1]
            self.padding += label_height // 2 + 1
        self.background_label = self.parameters.get('background_label', BACKGROUND_LABEL)

        self.text, content_width, content_height = self.format_text(text)
        self.is_empty_text = len(self.text) == 0

        self.rotated_text = self.text_type == 'word' and len(self.text) > 2
        if self.rotated_text:
            hypo = np.sqrt((content_width**2 + content_height**2) / 4)
            shift = np.arctan(content_height / content_width)
            actual_max_rot = np.arcsin((self.height / 2) / hypo) if hypo > self.height / 2 else np.inf
            actual_max_rot = (actual_max_rot - shift) * 180 / np.pi
            min_rot, max_rot = TEXT_ROTATION_ANGLE_RANGE
            min_rot, max_rot = max(min_rot, -actual_max_rot), min(max_rot, actual_max_rot)
            self.rotation_angle = uniform(min_rot, max_rot)
            shift = -shift if self.rotation_angle < 0 else shift
            new_content_height = 2 * abs(round(float(np.sin((self.rotation_angle * np.pi / 180) + shift) * hypo)))
            self.rot_padding = (new_content_height - content_height) // 2
            self.content_width, self.content_height = content_width, new_content_height
            self.pos_x = randint(0, max(0, self.width - self.content_width))
            self.pos_y = randint(self.rot_padding, max(self.rot_padding, self.height - self.content_height))
        else:
            self.content_width, self.content_height = content_width, content_height
            self.pos_x = randint(0, max(0, self.width - self.content_width))
            self.pos_y = randint(0, max(0, self.height - self.content_height))

    def format_text(self, text):
        if self.font_type in ['normal', 'handwritten']:
            text = unidecode(text)
        elif self.font_type == 'arabic':
            text = google(text, src='en', dst='ar')
            text = get_display(arabic_reshaper.reshape(text))
        elif self.font_type == 'chinese':
            text = google(text, src='en', dst='zh-CN')
        else:
            raise NotImplementedError

        width, height = self.width - 2 * self.padding, self.height - 2 * self.padding
        text = (text.upper() if self.uppercase else text).strip()
        if self.text_type == 'word':
            word_as_number = choice([True, False])
            if word_as_number:
                n_letter = randint(1, 5)
                result_text = str(randint(0, 10**n_letter - 1))
            else:
                words = text.split(' ')
                result_text = rand_choice(words)
                iteration = 1
                while (not str.isalnum(result_text) or len(result_text) < 1) and iteration < 40:
                    result_text = rand_choice(words)
                    iteration += 1
                if not str.isalnum(result_text) or len(result_text) < 1:
                    result_text = words[0][:randint(4, 10)]
                line_width = self.font.getsize(result_text)[0]
                while line_width > width and len(result_text) > 2:
                    result_text = result_text[:-1]
                    line_width = self.font.getsize(result_text)[0]

        else:
            max_lines = 1 if self.uniline else self.n_max_lines
            result_text, lines = '', ''
            text_height, cur_idx, n_lines = 0, 0, -1
            while text_height <= height:
                result_text = lines
                n_lines += 1
                line = text[cur_idx:].lstrip()
                cur_idx += len(text[cur_idx:]) - len(line)  # adjust cur_idx if stripped
                if len(line) == 0 or n_lines == max_lines:
                    break
                line_width = self.font.getsize(line)[0]
                avg_char_width = line_width / len(line)
                if line_width > width:
                    index = int(width / avg_char_width) + 10  # take larger slice in case of small characters
                    cut = max(line[:index].rfind(' '), line.find(' '))  # in case no space found in slice (small width)
                    line = line[:cut].strip()
                    line_width = self.font.getsize(line)[0]
                while line_width > width:
                    if ' ' in line:  # remove word by word
                        line = line[:line.rfind(' ')].strip()
                    else:  # remove character by character
                        line = line[:-1]
                    line_width = self.font.getsize(line)[0]

                cur_idx += len(line) + 1
                if self.justified:
                    w_space = self.font.getsize(' ')[0]
                    n_spaces = line.count(' ')
                    n_spaces_to_add = (width - line_width) // w_space
                    if n_spaces > 0 and n_spaces_to_add > 0:
                        q, r = n_spaces_to_add // n_spaces + 1, n_spaces_to_add % n_spaces
                        if q < 5:
                            if q > 1:
                                line = line.replace(' ', q * ' ')
                            pos = 0
                            while r > 0:
                                space_idx = line[pos:].find(' ') + pos
                                line = line[:space_idx] + ' ' + line[space_idx:]
                                pos = space_idx + q + 1
                                r -= 1
                lines = '{}\n{}'.format(lines, line) if lines else line
                text_height = self.font.getsize_multiline(lines, spacing=self.spacing)[1]

        if '\n' in result_text and self.justified:  # we dont want to justify the last line
            result_text, last_line = result_text.rsplit('\n', 1)
            last_line = ' '.join(last_line.split())
            result_text = '{}\n{}'.format(result_text, last_line)

        content_width, content_height = self.font.getsize_multiline(result_text, spacing=self.spacing)
        content_width += 2 * self.padding
        content_height += 2 * self.padding

        return result_text, content_width, content_height

    def to_image(self):
        canvas = Image.new('RGBA', self.size)
        image_draw = ImageDraw.Draw(canvas)
        if self.is_empty_text:
            return canvas

        if self.with_bbox:
            x, y = self.pos_x, self.pos_y
            p = self.bbox_params['width'] // 2 + 1
            image_draw.rectangle([(x+p, y+p), (x+self.content_width-p, y+self.content_height-p)], **self.bbox_params)

        if self.underlined:
            x, y = self.pos_x + self.padding, self.pos_y + self.padding + self.underline_padding
            line_height = self.font.getsize('A')[1]
            ascent, descent = self.font.getmetrics()
            lines = self.text.split('\n')
            for k in range(len(lines)):
                image_draw.line((x, y + ascent, x + self.content_width - 2 * self.padding, y + ascent),
                                **self.underline_params)
                y += line_height + self.spacing

        image_draw.text((self.pos_x + self.padding, self.pos_y + self.padding), self.text, self.colors_alpha,
                        font=self.font, spacing=self.spacing)

        if self.rotated_text:
            x, y = self.pos_x, self.pos_y
            img = canvas.crop((x, y - self.rot_padding, x + self.content_width, y + self.content_height -
                               self.rot_padding))
            img = img.rotate(self.rotation_angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            canvas.paste(img, (self.pos_x, self.pos_y - self.rot_padding))
        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))
        if self.transpose:
            canvas = canvas.transpose(Image.ROTATE_90)

        return canvas

    def to_label_as_array(self):
        label = np.full(self.size, self.background_label, dtype=np.uint8)
        if not self.as_negative and len(self.text) > 0:
            x, y = self.pos_x + self.padding, self.pos_y + self.padding
            line_height = self.font.getsize('A')[1]
            if self.baseline_as_label:
                label_height = TEXT_BASELINE_HEIGHT // 2
            else:
                if self.text.isdigit():
                    char = '1'
                elif self.uppercase:
                    char = 'A'
                else:
                    char = 'a'
                label_height = self.font.font.getsize(char)[0][1]
            ascent, descent = self.font.getmetrics()
            offset_y = max(0, ascent - label_height)
            if self.baseline_as_label:
                ascent += label_height + 1
            lines = self.text.split('\n')
            if self.with_border_label:
                border = label_height // 2 if not self.baseline_as_label else TEXT_BASELINE_HEIGHT // 2 + 1
                for line in lines:
                    if len(line) == 0:
                        continue
                    line_width = self.font.getsize(line)[0]
                    x_min, x_max = max(0, x - border), min(x + line_width + border, label.shape[0])
                    y_min, y_max = max(0, y + offset_y - border), min(y + ascent + border, label.shape[1])
                    label[x_min:x_max, y_min:y_max] = self.border_label
                    y += line_height + self.spacing

            x, y = self.pos_x + self.padding, self.pos_y + self.padding
            for line in lines:
                line_width = self.font.getsize(line)[0]
                y_min, y_max = y + offset_y, min(y + ascent, label.shape[1])
                label[x:x+line_width, y_min:y_max] = self.label
                y += line_height + self.spacing

        label = label.transpose()
        if self.rotated_text:
            center = (self.pos_x + self.content_width / 2, self.pos_y + self.content_height / 2 - self.rot_padding)
            R = cv2.getRotationMatrix2D(center, self.rotation_angle, 1)
            label = cv2.warpAffine(label, R, self.size, flags=cv2.INTER_NEAREST, borderValue=self.background_label)
        if self.transpose:
            return np.rot90(label)
        else:
            return label

    def to_label_as_img(self):
        arr = self.to_label_as_array()
        res = np.full(arr.shape + (3,), self.background_label, dtype=np.uint8)
        res[arr == self.label] = self.color
        if self.with_border_label:
            res[arr == self.border_label] = self.border_color
        return Image.fromarray(res)


class CaptionElement(AbstractTextElement):
    label = CAPTION_LABEL
    color = CAPTION_COLOR
    text_type = 'caption'
    n_max_lines = 3
    n_min_characters = 50
    font_size_range = (20, 60)
    line_spacing_range = (1, 8)


class ParagraphElement(AbstractTextElement):
    label = PARAGRAPH_LABEL
    color = PARAGRAPH_COLOR
    text_type = 'paragraph'
    n_max_lines = 1000
    n_min_characters = 400
    font_size_range = (20, 60)
    line_spacing_range = (1, 10)


class TitleElement(AbstractTextElement):
    label = TITLE_LABEL
    color = TITLE_COLOR
    text_type = 'title'
    n_max_lines = 20
    n_min_characters = 50
    font_size_range = (50, 150)
    line_spacing_range = (5, 50)


class WordElement(AbstractTextElement):
    label = FLOATING_WORD_LABEL
    color = FLOATING_WORD_COLOR
    text_type = 'word'
    n_max_lines = 1
    n_min_characters = 100
    font_size_range = (20, 60)
    line_spacing_range = (1, 1)


class TableElement(AbstractElement):
    label = TABLE_WORD_LABEL
    color = TABLE_WORD_COLOR
    border_label = TEXT_BORDER_LABEL
    border_color = TEXT_BORDER_COLOR
    font_size_range = (20, 50)
    name = 'table'

    @use_seed()
    def generate_content(self):
        min_fs, max_fs = self.font_size_range
        self.font_path = self.parameters.get('font_path') or AbstractTextElement.get_random_font()
        rescaled_height = (self.height * 2) // 3  # to avoid oversized letters
        actual_max_fs = min(rescaled_height, max_fs)
        if min_fs < actual_max_fs:
            self.font_size = randint(min_fs, actual_max_fs)
        else:
            self.font_size = actual_max_fs

        self.baseline_as_label = self.parameters.get('baseline_as_label', False)
        if self.baseline_as_label:
            self.label, self.color = BASELINE_LABEL, BASELINE_COLOR

        self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        self.as_negative = self.parameters.get('as_negative', False)
        self.blur_radius = uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        self.opacity = randint(*NEG_ELEMENT_OPACITY_RANGE[self.name] if self.as_negative
                               else POS_ELEMENT_OPACITY_RANGE[self.name])
        self.colored = choice([True, False], p=[TEXT_COLORED_FREQ, 1 - TEXT_COLORED_FREQ])
        self.colors = tuple([randint(0, 100)] * 3) if not self.colored else tuple([randint(0, 100) for _ in range(3)])
        self.colors_alpha = self.colors + (self.opacity,)

        self.padding = 0
        self.with_border_label = self.parameters.get('with_border_label', False)
        if self.with_border_label:
            label_height = self.font.font.getsize('A')[0][1]
            border_label_size = label_height // 2 + 1
            self.padding += border_label_size
        self.line_params = {
            'width': randint(*LINE_WIDTH_RANGE),
            'fill': tuple([randint(0, 100)] * 3) + (self.opacity,),
        }
        self.column_params = {
            'width': randint(*LINE_WIDTH_RANGE),
            'fill': tuple([randint(0, 100)] * 3) + (self.opacity,),
        }

        if 'text' in self.parameters:
            text = self.parameters['text']
        else:
            n_char = 0
            while (n_char <= ParagraphElement.n_min_characters):
                self.text_path = choice(DATABASE[TEXT_RESRC_NAME])
                with open(self.text_path) as f:
                    text = f.read().rstrip('\n')
                n_char = len(text)
        dictionary = text.split(' ')

        self.table, self.content_width, self.content_height = self._generate_table(dictionary)
        self.pos_x = randint(0, max(0, self.width - self.content_width))
        self.pos_y = randint(0, max(0, self.height - self.content_height))

    def _generate_table(self, dictionary):
        width, height = randint(min(200, self.width), self.width), randint(min(200, self.height), self.height)

        line_size_min = round(self.font_size * 1.3)
        line_size_max = round(self.font_size * 2.5)
        lines = np.cumsum(np.random.randint(line_size_min, line_size_max, 40))
        lines = lines[lines < height - line_size_min].tolist()
        columns = np.cumsum(np.random.randint(*TABLE_LAYOUT_RANGE['col_size_range'], 20))
        columns = columns[columns < width - TABLE_LAYOUT_RANGE['col_size_range'][0]].tolist()

        words, word_positions = [], []
        for i, c in enumerate([0] + columns):
            for j, l in enumerate([0] + lines):
                word_as_number = choice([True, False])
                if word_as_number:
                    n_letter = randint(2, 9)
                    word = f'{randint(0, 10**n_letter - 1):,}'
                else:
                    word = rand_choice(dictionary)
                    uppercase = choice([True, False])
                    if uppercase:
                        word = word.upper()

                cell_width = columns[i] - c if i < len(columns) else width - c
                cell_height = lines[j] - l if j < len(lines) else height - l
                while self.font.getsize(word)[0] + 2 * self.padding > cell_width and len(word) > 0:
                    word = word[:-1].strip()
                if len(word) > 0:
                    w, h = self.font.getsize(word)
                    p_c, p_l = (cell_width - w) // 2, (cell_height - h) // 2
                    words.append(word)
                    word_positions.append((c + p_c, l + p_l))

        return ({'lines': lines, 'columns': columns, 'words': words, 'word_positions': word_positions}, width, height)

    def to_image(self):
        canvas = Image.new('RGBA', self.size)
        draw = ImageDraw.Draw(canvas)
        pos_x_width, pos_y_height = self.pos_x + self.content_width, self.pos_y + self.content_height
        for l in self.table['lines']:
            draw.line([self.pos_x, self.pos_y + l, pos_x_width, self.pos_y + l], **self.line_params)
        for c in self.table['columns']:
            draw.line([self.pos_x + c, self.pos_y, self.pos_x + c, pos_y_height], **self.column_params)

        for word, pos in zip(self.table['words'], self.table['word_positions']):
            pos = pos[0] + self.pos_x, pos[1] + self.pos_y
            draw.text(pos, word, font=self.font, fill=self.colors_alpha)

        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))
        return canvas

    def to_label_as_array(self):
        label = np.full(self.size, BACKGROUND_LABEL, dtype=np.uint8)
        if self.as_negative:
            return label.transpose()
        else:
            ascent, descent = self.font.getmetrics()
            if self.baseline_as_label:
                label_height = TEXT_BASELINE_HEIGHT // 2
                offset_y = ascent - label_height
                ascent += label_height + 1

            if self.with_border_label:
                for word, pos in zip(self.table['words'], self.table['word_positions']):
                    if len(word) == 0:
                        continue
                    x, y = self.pos_x + pos[0], self.pos_y + pos[1]
                    w = self.font.getsize(word)[0]
                    if not self.baseline_as_label:
                        if word.replace(',', '').isdigit():
                            char = '1'
                        elif word.isupper():
                            char = 'A'
                        else:
                            char = 'a'
                        label_height = self.font.font.getsize(char)[0][1]
                        offset_y = ascent - label_height
                    else:
                        label_height = TEXT_BASELINE_HEIGHT
                    border = label_height // 2 + 1
                    x_min, x_max = max(0, x-border), min(x + w + border, label.shape[0])
                    y_min, y_max = max(0, y + offset_y - border), min(y + ascent + border, label.shape[1])
                    label[x_min:x_max, y_min:y_max] = self.border_label

            for word, pos in zip(self.table['words'], self.table['word_positions']):
                if len(word) == 0:
                    continue
                x, y = self.pos_x + pos[0], self.pos_y + pos[1]
                w = self.font.getsize(word)[0]
                if not self.baseline_as_label:
                    if word.replace(',', '').isdigit():
                        char = '1'
                    elif word.isupper():
                        char = 'A'
                    else:
                        char = 'a'
                    label_height = self.font.font.getsize(char)[0][1]
                    offset_y = ascent - label_height
                label[x:x+w, y+offset_y:y+ascent] = self.label

            return label.transpose()

    def to_label_as_img(self):
        arr = self.to_label_as_array()
        res = np.zeros(arr.shape + (3,), dtype=np.uint8)
        res[arr == self.label] = self.color
        if self.with_border_label:
            res[arr == self.border_label] = self.border_color
        return Image.fromarray(res)
