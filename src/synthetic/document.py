import cv2
from PIL import Image, ImageFilter

import numpy as np
from numpy.random import choice, uniform, randint as np_randint
from random import randint

from utils import coerce_to_path_and_check_exist, use_seed
from utils.constant import (SEG_GROUND_TRUTH_FMT, LABEL_TO_COLOR_MAPPING, CONTEXT_BACKGROUND_LABEL,
                            ILLUSTRATION_COLOR, ILLUSTRATION_LABELS, TEXT_COLOR, TEXT_LABELS)
from utils.image import draw_line, paste_with_blured_borders, resize

from synthetic.element import (BackgroundElement, ContextBackgroundElement, DrawingElement, GlyphElement, ImageElement,
                               TitleElement, ParagraphElement, CaptionElement, WordElement, TableElement,
                               get_random_noise_pattern)


DEFAULT_DOCUMENT_WIDTH = 1192
DEFAULT_DOCUMENT_HEIGHT = 1684
DOCUMENT_HEIGHT_RANGE = (1192, 2176)

LAYOUT_RANGE = {
    'nb_h': (1, 2),
    'nb_v': (0, 4),
    'nb_v_lines': (0, 2),
    'nb_h_lines': (0, 2),
    'nb_noise_patterns': (0, 5),
    'nb_words': (0, 10),
    'margin_h': (5, 40),
    'margin_v': (5, 40),
    'padding_h': (5, 80),
    'padding_v': (5, 80),
    'caption_padding_v': (0, 20),
    'context_margin_h': (0, 300),
    'context_margin_v': (0, 200),
}

BACKGROUND_BLURED_BORDER_WIDTH_RANGE = (1, 10)
BLUR_RADIUS_RANGE = (0.2, 0.4)
LINE_OPACITY_RANGE = (100, 255)
LINE_STD_GAUSSIAN_NOISE_RANGE = (4, 40)
LINE_WIDTH_RANGE = (1, 4)

BLACK_AND_WHITE_FREQ = 0.5
COMMON_FONT_FREQ = 0.5
CONTEXT_BACKGROUND_FREQ = 0.3
DOUBLE_PAGE_FREQ = 0.3
DOUBLE_COLUMN_FREQ = 0.3

ELEMENT_FREQ = {
    DrawingElement: 0.1,
    GlyphElement: 0.1,
    ImageElement: 0.3,
    ParagraphElement: 0.3,
    TableElement: 0.05,
    TitleElement: 0.15,
}


class SyntheticDocument:
    available_elements = [DrawingElement, GlyphElement, ImageElement, ParagraphElement, TableElement, TitleElement]

    @use_seed()
    def __init__(self, width=DEFAULT_DOCUMENT_WIDTH, height=DEFAULT_DOCUMENT_HEIGHT, img_size=None,
                 text_border_label=True, baseline_as_label=False, merged_labels=True):
        if height is None:
            height = randint(*DOCUMENT_HEIGHT_RANGE)
        self.text_border_label = text_border_label
        self.baseline_as_label = baseline_as_label
        self.merged_labels = merged_labels

        double_page = choice([True, False], p=[DOUBLE_PAGE_FREQ, 1 - DOUBLE_PAGE_FREQ])
        if double_page:
            if img_size is not None:
                width, height = resize(Image.new('L', (2 * width, height)), img_size, keep_aspect_ratio=True).size
                self.width, self.height = width // 2, height
                self.document_width, self.document_height = width, height
            else:
                self.width, self.height = width, height
                self.document_width, self.document_height = 2 * width, height
            n_pages = 2
        else:
            if img_size is not None:
                width, height = resize(Image.new('L', (width, height)), img_size, keep_aspect_ratio=True).size
            self.width, self.height = width, height
            self.document_width, self.document_height = width, height
            n_pages = 1

        self.blur_radius = uniform(*BLUR_RADIUS_RANGE)
        self.black_and_white = choice([True, False], p=[BLACK_AND_WHITE_FREQ, 1 - BLACK_AND_WHITE_FREQ])
        self.add_context_background = choice([True, False], p=[CONTEXT_BACKGROUND_FREQ, 1 - CONTEXT_BACKGROUND_FREQ])
        self.backgrounds = []
        if self.add_context_background:
            margins = {
                'left': randint(*LAYOUT_RANGE['context_margin_h']),
                'right': randint(*LAYOUT_RANGE['context_margin_h']),
                'top': randint(*LAYOUT_RANGE['context_margin_h']),
                'bottom': randint(*LAYOUT_RANGE['context_margin_v']),
            }
            self.context_background = {
                'element': ContextBackgroundElement(self.document_width, self.document_height),
                'margins': margins
            }

            width = (self.document_width - margins['left'] - margins['right']) // n_pages
            height = self.document_height - margins['top'] - margins['bottom']
            background_element = BackgroundElement(width, height)
            for k in range(n_pages):
                self.backgrounds.append({
                    'element': background_element,
                    'position': (margins['left'] if k == 0 else margins['left'] + width, margins['top']),
                    'border_width': randint(*BACKGROUND_BLURED_BORDER_WIDTH_RANGE),
                })
        else:
            self.context_background = None
            background_element = BackgroundElement(self.width, self.height)
            for k in range(n_pages):
                self.backgrounds.append({
                    'element': background_element,
                    'position': (0 if k == 0 else self.width, 0),
                    'border_width': 0,
                })

        self.lines = self._generate_random_lines()
        self.noise_patterns = self._generate_random_noise_patterns()
        self.elements, self.positions = self._generate_random_layout()
        self.neg_elements, self.neg_positions = self._generate_random_layout(as_negative=True)

    @property
    def size(self):
        return (self.document_width, self.document_height)

    @use_seed()
    def _generate_random_lines(self):
        lines = []
        for background in self.backgrounds:
            bg_width, bg_height = background['element'].size
            bg_x, bg_y = background['position']
            for _ in range(randint(*LAYOUT_RANGE['nb_v_lines'])):
                params = {
                    'position': list(zip(*(np.sort(np_randint(bg_x, bg_x + bg_width + 1, 2)),
                                           [randint(bg_y, bg_y + bg_height)] * 2))),
                    'color': tuple([randint(0, 50)] * 3) + (randint(*LINE_OPACITY_RANGE),),
                    'width': randint(*LINE_WIDTH_RANGE),
                    'blur_radius': uniform(*LINE_WIDTH_RANGE),
                    'std_gaussian_noise': tuple([randint(*LINE_STD_GAUSSIAN_NOISE_RANGE) for _ in range(3)]),
                }
                lines.append(params)

            for _ in range(randint(*LAYOUT_RANGE['nb_h_lines'])):
                params = {
                    'position': list(zip(*([randint(bg_y, bg_y + bg_height)] * 2,
                                           np.sort(np_randint(bg_x, bg_x + bg_width + 1, 2))))),
                    'color': tuple([randint(0, 50)] * 3) + (randint(*LINE_OPACITY_RANGE),),
                    'width': randint(*LINE_WIDTH_RANGE),
                    'blur_radius': uniform(*LINE_WIDTH_RANGE),
                    'std_gaussian_noise': tuple([randint(*LINE_STD_GAUSSIAN_NOISE_RANGE) for _ in range(3)]),
                }
                lines.append(params)

        return lines

    @use_seed()
    def _generate_random_noise_patterns(self):
        patterns, positions = [], []
        for background in self.backgrounds:
            bg_width, bg_height = background['element'].size
            bg_x, bg_y = background['position']
            for _ in range(randint(*LAYOUT_RANGE['nb_noise_patterns'])):
                pattern, hue_color, value_ratio, position = get_random_noise_pattern(bg_width, bg_height)
                position = (position[0] + bg_x, position[1] + bg_y)
                patterns.append((pattern, hue_color, value_ratio))
                positions.append(position)

        return patterns, positions

    @use_seed()
    def _generate_random_layout(self, as_negative=False):
        common_font = choice([True, False], p=[COMMON_FONT_FREQ, 1 - COMMON_FONT_FREQ])
        if common_font:
            font_path = ParagraphElement.get_random_font()
        else:
            font_path = None
        element_kwargs = {'as_negative': as_negative, 'font_path': font_path, 'with_border_label': not as_negative and
                          self.text_border_label, 'baseline_as_label': self.baseline_as_label}

        elements, positions = [], []
        if self.add_context_background:
            is_light_bg = self.context_background['element'].intensity > 100
            margins = self.context_background['margins']
            caption_padding = randint(*LAYOUT_RANGE['caption_padding_v'])
            horizontal = choice([True, False])
            element = None
            if horizontal:
                text_height = min(margins['bottom'] - caption_padding, (CaptionElement.font_size_range[1] * 3) // 2)
                max_font_size = (text_height * 2) // 3
                if max_font_size >= CaptionElement.font_size_range[0]:
                    width = self.document_width - margins['left'] - margins['right']
                    element = CaptionElement(width, text_height, dark_mode=is_light_bg,
                                             background_label=CONTEXT_BACKGROUND_LABEL, **element_kwargs)
                    position = (margins['left'], self.document_height - margins['bottom'] + caption_padding)
            else:
                text_height = min(margins['left'] - caption_padding, (CaptionElement.font_size_range[1] * 3) // 2)
                if (text_height * 2) // 3 >= CaptionElement.font_size_range[0]:
                    height = self.document_height - margins['top'] - margins['bottom']
                    element = CaptionElement(height, text_height, transpose=True,
                                             dark_mode=is_light_bg, background_label=CONTEXT_BACKGROUND_LABEL,
                                             **element_kwargs)
                    position = (caption_padding, margins['top'])
            if isinstance(element, CaptionElement) and len(element.text) > 0:
                elements.append(element)
                positions.append(position)

        weights = np.array([ELEMENT_FREQ[e] for e in self.available_elements])
        weights /= weights.sum()
        nb_words = randint(*LAYOUT_RANGE['nb_words'])
        for i, background in enumerate(self.backgrounds):
            bg_width, bg_height = background['element'].size
            bg_x, bg_y = background['position']
            bg_width -= background['element'].inherent_left_margin
            if i == 0:
                bg_x += background['element'].inherent_left_margin

            nb_v_elements = randint(*LAYOUT_RANGE['nb_v'])
            if nb_v_elements > 0:
                margin_v = randint(*LAYOUT_RANGE['margin_v'])
                padding_v = randint(*LAYOUT_RANGE['padding_v'])
                height = (bg_height - 2 * margin_v - (nb_v_elements - 1) * padding_v) // nb_v_elements
                cur_y = bg_y + margin_v

                for _ in range(nb_v_elements):
                    nb_h_elements = choice(LAYOUT_RANGE['nb_h'], p=[1-DOUBLE_COLUMN_FREQ, DOUBLE_COLUMN_FREQ])
                    margin_h = randint(*LAYOUT_RANGE['margin_h'])
                    padding_h = randint(*LAYOUT_RANGE['padding_h'])
                    width = (bg_width - 2 * margin_h - (nb_h_elements - 1) * padding_h) // nb_h_elements
                    layout_x_pos = [bg_x]
                    for k in range(nb_h_elements):
                        element = choice(self.available_elements, p=weights)(width, height, **element_kwargs)
                        while nb_h_elements == 2 and isinstance(element, TitleElement):
                            element = choice(self.available_elements, p=weights)(width, height, **element_kwargs)
                        position = (bg_x + margin_h + k * padding_h + k * width, cur_y)
                        elements.append(element)
                        positions.append(position)
                        absolute_element_x = position[0] + element.position[0]
                        layout_x_pos += [absolute_element_x, absolute_element_x + element.content_width]

                        if isinstance(element, (DrawingElement, ImageElement)):
                            content_height = element.content_height + element.position[1]
                            caption_padding = randint(*LAYOUT_RANGE['caption_padding_v'])
                            text_height = min(height - (content_height + caption_padding),
                                              (CaptionElement.font_size_range[1] * 3) // 2)
                            max_font_size = (text_height * 2) // 3
                            if max_font_size >= CaptionElement.font_size_range[0]:
                                element = CaptionElement(width, text_height, **element_kwargs)
                                position = (position[0], cur_y + content_height + caption_padding)
                            else:
                                content_width = element.content_width + element.position[0]
                                caption_padding = randint(*LAYOUT_RANGE['caption_padding_v'])
                                text_height = min(width - (content_width + caption_padding),
                                                  (CaptionElement.font_size_range[1] * 3) // 2)
                                max_font_size = (text_height * 2) // 3
                                if max_font_size >= CaptionElement.font_size_range[0]:
                                    element = CaptionElement(height, text_height, transpose=True, **element_kwargs)
                                    position = (position[0] + content_width + caption_padding, cur_y)
                                    absolute_element_x = position[0] + element.position[1]
                                    layout_x_pos += [absolute_element_x, absolute_element_x + element.content_height]
                            if isinstance(element, CaptionElement) and len(element.text) > 0:
                                elements.append(element)
                                positions.append(position)

                    if not as_negative and nb_words > 0:
                        layout_x_pos.append(bg_width)
                        h_spaces = np.diff(layout_x_pos)[::2]
                        for k, h_space in enumerate(h_spaces):
                            word_element = WordElement(h_space, height, **element_kwargs)
                            if word_element.content_width < h_space:
                                elements.append(word_element)
                                positions.append((layout_x_pos[k*2], cur_y))
                                nb_words -= 1
                                if nb_words == 0:
                                    break

                    cur_y += height + padding_v

        return elements, positions

    def to_image(self):
        if self.add_context_background:
            canvas = self.context_background['element'].to_image()
        else:
            canvas = Image.new(mode='RGB', size=self.size)

        for k, background in enumerate(self.backgrounds):
            background_img = background['element'].to_image(flip=(k == 1))
            paste_with_blured_borders(canvas, background_img, background['position'], background['border_width'])

        # Elements
        mean_background_color = background_img.resize((1, 1)).getpixel((0, 0))
        self.draw_elements(canvas, canvas_color=mean_background_color, as_negative=True)
        self.draw_elements(canvas, canvas_color=mean_background_color)

        # Noise
        [draw_line(canvas, **kwargs) for kwargs in self.lines]
        self.draw_noise_patterns(canvas)

        if self.blur_radius > 0:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        if self.black_and_white:
            canvas = canvas.convert("L").convert("RGB")

        return canvas

    def draw_elements(self, canvas, canvas_color, as_negative=False):
        elements_positions = (self.neg_elements, self.neg_positions) if as_negative else (self.elements, self.positions)
        for element, position in zip(*elements_positions):
            if isinstance(element, ImageElement):
                img = element.to_image(canvas_color=canvas_color)
            elif isinstance(element, DrawingElement):
                x, y = position
                width, height = element.size
                img = element.to_image(canvas=canvas.crop((x, y, x+width, y+height)))
            else:
                img = element.to_image()

            if as_negative:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            canvas.paste(img, position, img)

    def draw_noise_patterns(self, canvas):
        for (noise, hue_color, value_ratio), pos in zip(*self.noise_patterns):
            x, y = pos
            width, height = noise.size
            patch = np.array(canvas.crop([x, y, x + width, y + height]))
            patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            patch_hsv[:, :, 0] = hue_color
            patch_hsv[:, :, 2] = patch_hsv[:, :, 2] * value_ratio
            new_patch = Image.fromarray(cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2RGB))
            canvas.paste(new_patch, pos, mask=noise)

    def to_image_as_array(self):
        return np.array(self.to_image(), dtype=np.float32) / 255

    def to_label_as_array(self, restricted_labels=None):
        if restricted_labels is None:
            restricted_labels = set(LABEL_TO_COLOR_MAPPING.keys())
        label = np.zeros(self.size[::-1], dtype=np.uint8)

        for background in self.backgrounds:
            arr = background['element'].to_label_as_array()
            x, y = background['position'][::-1]
            label[x:x+background['element'].height, y:y+background['element'].width] = arr

        for element, position in zip(self.elements, self.positions):
            if element.label in restricted_labels:
                arr = element.to_label_as_array()
                x, y = position[::-1]
                if hasattr(element, 'transpose') and element.transpose:
                    label[x:x+element.width, y:y+element.height] = arr
                else:
                    label[x:x+element.height, y:y+element.width] = arr

        return label

    def to_label_as_img(self):
        arr = self.to_label_as_array()
        res = np.zeros(arr.shape + (3,), dtype=np.uint8)
        for label, color in LABEL_TO_COLOR_MAPPING.items():
            if self.merged_labels:
                if label in ILLUSTRATION_LABELS:
                    res[arr == label] = ILLUSTRATION_COLOR
                elif label in TEXT_LABELS:
                    res[arr == label] = TEXT_COLOR
                else:
                    res[arr == label] = color
            else:
                res[arr == label] = color
        return Image.fromarray(res)

    def save(self, name, output_dir):
        output_dir = coerce_to_path_and_check_exist(output_dir)
        self._save_image('{}.jpg'.format(name), output_dir)
        self._save_label(SEG_GROUND_TRUTH_FMT.format(name, 'png'), output_dir)

    def _save_image(self, name, output_dir):
        img = self.to_image()
        img.save(output_dir / name)

    def _save_label(self, name, output_dir):
        img = self.to_label_as_img()
        img.save(output_dir / name)
