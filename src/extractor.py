import argparse
import cv2
from PIL import Image, ImageDraw

import numpy as np
import torch

from models import load_model_from_path
from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir, get_files_from_dir
from utils.constant import BACKGROUND_LABEL, ILLUSTRATION_LABEL, TEXT_LABEL, LABEL_TO_COLOR_MAPPING, MODEL_FILE
from utils.image import Pdf2Image, resize
from utils.logger import get_logger, print_info, print_error
from utils.path import MODELS_PATH

VALID_EXTENSIONS = ['jpeg', 'JPEG', 'jpg', 'JPG', 'pdf', 'tiff']

ADDITIONAL_MARGIN_RATIO = {
    ILLUSTRATION_LABEL: 0,
    TEXT_LABEL: 0.25,
}
LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD = {
    ILLUSTRATION_LABEL: 0.005,
    TEXT_LABEL: 0.0001,
}
LABEL_TO_NAME = {
    ILLUSTRATION_LABEL: 'illustration',
    TEXT_LABEL: 'text',
}


class Extractor:
    """
    Extract elements from files in a given input_dir folder and save them in the provided output_dir.
    Supported input extensions are: jpg, png, tiff, pdf.
    """

    def __init__(self, input_dir, output_dir, labels_to_extract=None, in_ext=VALID_EXTENSIONS, out_ext='jpg',
                 tag='default', save_annotations=True, straight_bbox=False, add_margin=True, draw_margin=False,
                 verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir).absolute()
        self.files = get_files_from_dir(self.input_dir, valid_extensions=in_ext, recursive=True, sort=True)
        self.output_dir = coerce_to_path_and_create_dir(output_dir).absolute()
        self.out_extension = out_ext
        self.logger = get_logger(self.output_dir, name='extractor')
        model_path = coerce_to_path_and_check_exist(MODELS_PATH / tag / MODEL_FILE)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, (self.img_size, restricted_labels, self.normalize) = load_model_from_path(
            model_path, device=self.device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
        self.model.eval()

        self.restricted_labels = sorted(restricted_labels)
        self.labels_to_extract = [1, 4] if labels_to_extract is None else sorted(labels_to_extract)
        if not set(self.labels_to_extract).issubset(self.restricted_labels):
            raise ValueError('Incompatible `labels_to_extract` and `tag` arguments: '
                             f'model was trained using {self.restricted_labels} labels only')

        self.save_annotations = save_annotations
        self.straight_bbox = straight_bbox
        self.add_margin = add_margin
        self.draw_margin = add_margin and draw_margin
        self.verbose = verbose
        self.print_and_log_info('Extractor initialised with kwargs {}'.format(
            {'tag': tag, 'labels_to_extract': self.labels_to_extract, 'save_annotations': save_annotations,
             'straight_bbox': straight_bbox, 'add_margin': add_margin, 'draw_margin': draw_margin}))
        self.print_and_log_info('Model characteristics: train_resolution={}, restricted_labels={}'
                                .format(self.img_size, self.restricted_labels))
        self.print_and_log_info('Found {} input files to process'.format(len(self.files)))

    def print_and_log_info(self, string):
        self.logger.info(string)
        if self.verbose:
            print_info(string)

    def print_and_log_error(self, string):
        self.logger.error(string)
        if self.verbose:
            print_error(string)

    def run(self):
        for filename in self.files:
            self.print_and_log_info('Processing {}'.format(filename.relative_to(self.input_dir)))
            try:
                imgs_with_names, output_path = self.get_images_and_output_path(filename)
            except (NotImplementedError, OSError) as e:
                self.print_and_log_error(e)
                imgs_with_names, output_path = [], None

            for img, name in imgs_with_names:
                img_with_annotations = img.copy()
                pred = self.predict(img)
                for label in self.labels_to_extract:
                    if label != BACKGROUND_LABEL:
                        extracted_elements = self.extract(img, pred, label, img_with_annotations)
                        path = output_path if len(self.labels_to_extract) == 1 else output_path / LABEL_TO_NAME[label]
                        for k, extracted_element in enumerate(extracted_elements):
                            extracted_element.save(path / '{}_{}.{}'
                                                   .format(name, k, self.out_extension))
                if self.save_annotations:
                    (output_path / 'annotation').mkdir(exist_ok=True)
                    img_with_annotations.save(output_path / 'annotation' / '{}_annotated.{}'
                                              .format(name, self.out_extension))

        self.print_and_log_info('Extractor run is over')

    def get_images_and_output_path(self, filename):
        if filename.suffix in ['.jpeg', '.JPEG', '.jpg', '.JPG', '.png', '.tiff']:
            imgs, names = [Image.open(filename).convert('RGB')], [filename.stem]
            output_path = self.output_dir / filename.parent.relative_to(self.input_dir)
            output_path.mkdir(exist_ok=True, parents=True)
        elif filename.suffix == '.pdf':
            self.print_and_log_info('Converting pdf to jpg')
            imgs = Pdf2Image.convert(filename)
            names = ['{}_p{}'.format(filename.stem, k + 1) for k in range(len(imgs))]
            output_path = self.output_dir / filename.parent.relative_to(self.input_dir) / filename.stem
            output_path.mkdir(exist_ok=True, parents=True)
        else:
            raise NotImplementedError('"{}" extension is currently not supported'.format(filename.suffix[1:]))

        if len(self.labels_to_extract) > 1:
            for label in self.labels_to_extract:
                if label != BACKGROUND_LABEL:
                    (output_path / LABEL_TO_NAME[label]).mkdir(exist_ok=True)

        return zip(imgs, names), output_path

    def predict(self, image):
        red_img = resize(image, size=self.img_size, keep_aspect_ratio=True)
        inp = np.array(red_img, dtype=np.float32) / 255
        if self.normalize:
            inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(self.device)  # HWC -> CHW tensor
        with torch.no_grad():
            pred = self.model(inp.reshape(1, *inp.shape))[0].max(0)[1].cpu().numpy()
        return pred

    def extract(self, image, pred, label, image_with_annotations):
        label_idx = self.restricted_labels.index(label) + 1
        color = LABEL_TO_COLOR_MAPPING[label]
        mask_pred = cv2.resize((pred == label_idx).astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
        _, contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        y_coord = []
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if self.save_annotations:
                draw = ImageDraw.Draw(image_with_annotations)
                draw.line(list(map(tuple, cnt.reshape(-1, 2).tolist())) + cnt[0][0].tolist(), fill=color, width=5)

            if cnt_area / (image.size[0] * image.size[1]) >= LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD[label]:
                if self.straight_bbox:
                    x, y, width, height = cv2.boundingRect(cnt)
                    if self.add_margin:
                        m = int(min(ADDITIONAL_MARGIN_RATIO[label] * width, ADDITIONAL_MARGIN_RATIO[label] * height))
                        bbox = np.asarray([[x-m, y-m], [x+width+m, y-m], [x+width+m, y+height+m], [x-m, y+height+m]])
                        bbox = np.clip(bbox, a_min=(0, 0), a_max=image.size)
                        margins = np.array([min(m, x), min(m, y), -min(image.size[0] - x - width, m),
                                            -min(image.size[1] - y - height, m)], dtype=np.int32)
                    else:
                        bbox = np.asarray([[x, y], [x+width, y], [x+width, y+height], [x, y+height]])
                    result_img = image.crop(tuple(bbox[0]) + tuple(bbox[2]))

                else:
                    rect = cv2.minAreaRect(cnt)
                    width, height, angle = int(rect[1][0]), int(rect[1][1]), rect[-1]
                    if self.add_margin:
                        m = int(min(ADDITIONAL_MARGIN_RATIO[label] * width, ADDITIONAL_MARGIN_RATIO[label] * height))
                        width, height = width + 2 * m, height + 2 * m
                        rect = (rect[0], (width, height), angle)
                        margins = np.array([m, m, -m, -m], dtype=np.int32)
                    bbox = np.int32(cv2.boxPoints(rect))
                    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(bbox.astype(np.float32), dst_pts)
                    result_img = Image.fromarray(cv2.warpPerspective(np.array(image), M, (width, height)))
                    if angle < -45:
                        result_img = result_img.transpose(Image.ROTATE_90)

                if self.draw_margin:
                    width, height = result_img.size
                    lw = int(min([0.01 * width, 0.01 * height]))
                    draw = ImageDraw.Draw(result_img)
                    rect = np.array([0, 0, width, height], dtype=np.int32) + margins
                    draw.rectangle(rect.tolist(), fill=None, outline=(59, 178, 226), width=lw)

                results.append(result_img)
                y_coord.append(bbox[:, 1].min())

                if self.save_annotations:
                    lw = int(min([0.005 * image.size[0], 0.005 * image.size[1]]))
                    draw = ImageDraw.Draw(image_with_annotations)
                    draw.line(list(map(tuple, bbox.tolist())) + [tuple(bbox[0])], fill=(0, 255, 0), width=lw)

        self.print_and_log_info('Cropped {} {}s out of {} connected components found'
                                .format(len(results), LABEL_TO_NAME[label], len(contours)))

        return [results[i] for i in np.argsort(y_coord)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract elements from files in a given directory')
    parser.add_argument('-i', '--input_dir', nargs='?', type=str, required=True, help='Input directory')
    parser.add_argument('-o', '--output_dir', nargs='?', type=str, required=True, help='Output directory')
    parser.add_argument('-t', '--tag', nargs='?', type=str, default='default', help='Model tag to use')
    parser.add_argument('-l', '--labels', nargs='+', type=int, default=[1, 4], help='Labels to extract')
    parser.add_argument('-s', '--save_annot', action='store_true', help='Whether to save annotations')
    parser.add_argument('-sb', '--straight_bbox', action='store_true', help='Use straight bounding box only to'
                        'fit connected components found, instead of rotated ones')
    parser.add_argument('-dm', '--draw_margin', action='store_true', help='Draw the margins added, for visual purposes')
    args = parser.parse_args()

    input_dir = coerce_to_path_and_check_exist(args.input_dir)
    extractor = Extractor(input_dir, args.output_dir, labels_to_extract=args.labels, tag=args.tag,
                          save_annotations=args.save_annot, straight_bbox=args.straight_bbox,
                          draw_margin=args.draw_margin)
    extractor.run()
