import argparse
from collections import defaultdict
import cv2
from PIL import Image, ImageDraw

import numpy as np
import torch

from models import load_model_from_path
from utils import coerce_to_path_and_create_dir, coerce_to_path_and_check_exist, get_files_from_dir
from utils.constant import (CONTEXT_BACKGROUND_COLOR, ILLUSTRATION_LABEL, TEXT_LABEL, LABEL_TO_COLOR_MAPPING,
                            MODEL_FILE, SEG_GROUND_TRUTH_FMT)
from utils.image import resize, LabeledArray2Image
from utils.logger import get_logger, print_info, print_error, print_warning
from utils.metrics import RunningMetrics
from utils.path import MODELS_PATH


VALID_EXTENSIONS = ['jpeg', 'JPEG', 'jpg', 'JPG', 'pdf', 'tiff']
GT_COLOR = (0, 255, 0)

LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD = {
    ILLUSTRATION_LABEL: 0.005,
    TEXT_LABEL: 0.0001,
}


class Evaluator:
    """Pipeline to evaluate a given trained segmentation NN model on a given input_dir"""

    def __init__(self, input_dir, output_dir, tag="default", seg_fmt=SEG_GROUND_TRUTH_FMT, labels_to_eval=None,
                 save_annotations=True, labels_to_annot=None, predict_bbox=False, verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir).absolute()
        self.files = get_files_from_dir(self.input_dir, valid_extensions=VALID_EXTENSIONS, recursive=True, sort=True)
        self.output_dir = coerce_to_path_and_create_dir(output_dir).absolute()
        self.seg_fmt = seg_fmt
        self.logger = get_logger(self.output_dir, name='evaluator')
        model_path = coerce_to_path_and_check_exist(MODELS_PATH / tag / MODEL_FILE)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, (self.img_size, restricted_labels, self.normalize) = load_model_from_path(
            model_path, device=self.device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
        self.model.eval()

        self.restricted_labels = sorted(restricted_labels)
        self.labels_to_eval = [ILLUSTRATION_LABEL] if labels_to_eval is None else sorted(labels_to_eval)
        self.labels_to_rm = set(self.restricted_labels).difference(self.labels_to_eval)
        assert len(set(self.labels_to_eval).intersection(self.restricted_labels)) == len(self.labels_to_eval)

        self.restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in self.restricted_labels]
        self.label_idx_color_mapping = {self.restricted_labels.index(l) + 1: c
                                        for l, c in zip(self.restricted_labels, self.restricted_colors)}
        self.color_label_idx_mapping = {c: l for l, c in self.label_idx_color_mapping.items()}

        self.metrics = defaultdict(lambda : RunningMetrics(self.restricted_labels, self.labels_to_eval))
        self.save_annotations = save_annotations
        self.labels_to_annot = labels_to_annot or self.labels_to_eval
        self.predict_bbox = predict_bbox
        self.verbose = verbose

        self.print_and_log_info('Output dir: {}'.format(self.output_dir.absolute()))
        self.print_and_log_info('Evaluator initialised with kwargs {}'.format(
            {'labels_to_eval': self.labels_to_eval, 'save_annotations': save_annotations}))
        self.print_and_log_info('Model tag: {}'.format(model_path.parent.name))
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

    def print_and_log_warning(self, string):
        self.logger.warning(string)
        if self.verbose:
            print_warning(string)

    def run(self):
        for filename in self.files:
            self.print_and_log_info('Processing {}'.format(filename.relative_to(self.input_dir)))
            label_file = filename.parent / self.seg_fmt.format(filename.stem, 'png')
            dir_path = filename.parent.relative_to(self.input_dir)
            img = Image.open(filename).convert('RGB')
            pred = self.predict(img)
            if not label_file.exists():
                self.print_and_log_warning('Ground truth not found')
                gt = None
            else:
                if Image.open(label_file).size == img.size:
                    gt = self.encode_segmap(Image.open(label_file))
                    self.metrics[str(dir_path)].update(gt, pred)
                else:
                    self.print_and_log_error(filename.relative_to(self.input_dir))

            if self.save_annotations:
                output_path = self.output_dir / dir_path
                output_path.mkdir(exist_ok=True)
                pred_img = LabeledArray2Image.convert(pred, self.label_idx_color_mapping)
                mask = Image.fromarray((np.array(pred_img) == (0, 0, 0)).all(axis=-1).astype(np.uint8) * 127 + 128)
                blend_img = Image.composite(img, pred_img, mask)
                empty_pred, empty_gt = np.all(pred == 0), True
                lw = int(min([0.01 * img.size[0], 0.01 * img.size[1]]))
                if gt is not None:
                    for label in self.labels_to_eval:
                        if label in gt:
                            mask_gt = (gt == label).astype(np.uint8)
                            contours = cv2.findContours(mask_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
                            for cnt in contours:
                                empty_gt = False
                                draw = ImageDraw.Draw(blend_img)
                                draw.line(list(map(tuple, cnt.reshape(-1, 2).tolist())) + cnt[0][0].tolist(),
                                          fill=GT_COLOR, width=lw)
                if not empty_pred or not empty_gt:
                    blend_img = resize(blend_img.convert('RGB'), (1000, 1000))
                    blend_img.save(output_path / '{}.jpg'.format(filename.stem))

        self.save_metrics()
        self.print_and_log_info('Evaluator run is over')

    def encode_segmap(self, img):
        arr_segmap = np.array(img)
        unique_colors = set([color for size, color in img.getcolors()]).difference({CONTEXT_BACKGROUND_COLOR})

        label = np.zeros(arr_segmap.shape[:2], dtype=np.uint8)
        for color in unique_colors:
            if color in self.restricted_colors:
                mask = (arr_segmap == color).all(axis=-1)
                label[mask] = self.color_label_idx_mapping[color]

        return label

    @torch.no_grad()
    def predict(self, image):
        red_img = resize(image, size=self.img_size, keep_aspect_ratio=True)
        inp = np.array(red_img, dtype=np.float32) / 255
        if self.normalize:
            inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
        inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(self.device)  # HWC -> CHW tensor
        pred = self.model(inp.reshape(1, *inp.shape))[0].max(0)[1].cpu().numpy()

        res = np.zeros(pred.shape, dtype=np.uint8)
        for label in self.labels_to_annot:
            mask_pred = (pred == self.restricted_labels.index(label) + 1).astype(np.uint8)
            _, contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area / (pred.shape[0] * pred.shape[1]) >= LABEL_TO_DOCUMENT_CC_AREA_RATIO_THRESHOLD[label]:
                    if self.predict_bbox:
                        x, y, width, height = cv2.boundingRect(cnt)
                        bbox = np.asarray([[x, y], [x+width, y], [x+width, y+height], [x, y+height]])
                        cv2.fillPoly(res, [bbox], color=self.restricted_labels.index(label) + 1)
                    else:
                        cv2.fillPoly(res, [cnt], color=self.restricted_labels.index(label) + 1)

        res = cv2.resize(res, image.size, interpolation=cv2.INTER_NEAREST)
        return res

    def save_metrics(self):
        metric_names = next(iter(self.metrics.values())).names
        all_values = [[] for _ in range(len(metric_names))]
        with open(self.output_dir / 'metrics.tsv', mode='w') as f:
            f.write('dir_name\t{}\n'.format('\t'.join(metric_names)))
            for name, metrics in self.metrics.items():
                values = list(metrics.get().values())
                f.write('{}\t{}\n'.format(name, '\t'.join(map('{:.4f}'.format, values))))
                [all_values[k].append(v) for k, v in enumerate(values)]
            if len(self.metrics) > 1:
                mean_values = list(map(np.mean, all_values))
                f.write('{}\t{}\n'.format('average', '\t'.join(map('{:.4f}'.format, mean_values))))

        print_info('Metrics saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to evaluate a NN model on a given input_dir')
    parser.add_argument('-i', '--input_dir', nargs='?', type=str, required=True, help='Input directory')
    parser.add_argument('-o', '--output_dir', nargs='?', type=str, required=True, help='Output directory')
    parser.add_argument('-t', '--tag', nargs='?', type=str, default='default', help='Model tag to evaluate')
    parser.add_argument('-l', '--labels', nargs='+', type=int, default=[1], help='Labels to eval')
    parser.add_argument('-s', '--save_annot', action='store_true', help='Whether to save annotations')
    parser.add_argument('-lta', '--labels_to_annot', nargs='+', type=int, default=None, help='Labels to annotate')
    parser.add_argument('-b', '--pred_bbox', action='store_true', help='Whether to predict bounding boxes')
    args = parser.parse_args()

    input_dir = coerce_to_path_and_check_exist(args.input_dir)
    evaluator = Evaluator(input_dir, args.output_dir, tag=args.tag, labels_to_eval=args.labels,
                          save_annotations=args.save_annot if args.labels_to_annot is None else True,
                          labels_to_annot=args.labels_to_annot, predict_bbox=args.pred_bbox)
    evaluator.run()
