from random import randint

from utils import coerce_to_path_and_check_exist, use_seed
from utils.constant import OCR_GROUND_TRUTH_FMT
from synthetic.document import SyntheticDocument
from synthetic.element import ParagraphElement, TitleElement


DEFAULT_LINE_WIDTH = 512
DEFAULT_LINE_HEIGHT = 32
PADDING_RANGE = (0, 5)


class SyntheticLine:
    """Create synthetic line with ocr ground-truth"""
    def __init__(self, width=DEFAULT_LINE_WIDTH, height=DEFAULT_LINE_HEIGHT, seed=None):
        self.width, self.height = width, height
        self.document = SyntheticDocument(seed=seed)
        self.line_imgs, self.labels = self._get_line_imgs_and_labels(seed=seed)

    @use_seed()
    def _get_line_imgs_and_labels(self):
        document_image = self.document.to_image()
        line_imgs, labels = [], []
        for element, global_position in zip(self.document.elements, self.document.positions):
            if isinstance(element, (ParagraphElement, TitleElement)):
                x = global_position[0] + element.position[0] + element.padding
                y = global_position[1] + element.position[1] + element.padding
                line_height = element.font.getsize('A')[1]

                lines = element.text.split('\n')
                for k, line in enumerate(lines):
                    if len(line) > 0:
                        padding = randint(*PADDING_RANGE)
                        line_width = element.font.getsize(line)[0]
                        bbox = (x-padding, y-padding, x+line_width+padding, y+line_height+padding)
                        line_img = document_image.crop(bbox)
                        line_imgs.append(line_img)
                        labels.append(line)
                    y += line_height + element.spacing
        return line_imgs, labels

    @property
    def nb_lines(self):
        return len(self.line_imgs)

    def save(self, prefix_name, output_dir):
        output_dir = coerce_to_path_and_check_exist(output_dir)
        self._save_images(prefix_name, output_dir)
        self._save_labels(prefix_name, output_dir)

    def _save_images(self, prefix_name, output_dir):
        max_len_id = len(str(self.nb_lines - 1))
        for k, line in enumerate(self.line_imgs):
            line.save(output_dir / '{}_{}.jpg'.format(prefix_name, str(k).zfill(max_len_id)))

    def _save_labels(self, prefix_name, output_dir):
        max_len_id = len(str(self.nb_lines - 1))
        for k, label in enumerate(self.labels):
            name = '{}_{}'.format(prefix_name, str(k).zfill(max_len_id))
            with open(output_dir / OCR_GROUND_TRUTH_FMT.format(name, 'txt'), mode='w') as f:
                f.write(label)
