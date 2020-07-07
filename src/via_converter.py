import argparse
import json
from PIL import Image, ImageDraw

from utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from utils.constant import ILLUSTRATION_COLOR, SEG_GROUND_TRUTH_FMT
from utils.logger import print_info, print_error


class ViaJson2Image:
    """
    Convert VIA annotated regions into image files. Arg `input_dir` must include original images as well as the
    json file created through VIA software.
    """

    def __init__(self, input_dir, output_dir, json_file='via_region_data.json', out_ext='png', color=ILLUSTRATION_COLOR,
                 verbose=True):
        self.input_dir = coerce_to_path_and_check_exist(input_dir)
        self.annotations = self.load_json(self.input_dir / json_file)
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.out_ext = out_ext
        self.color = color
        self.mode = 'L' if isinstance(color, int) else 'RGB'
        self.background_color = 0 if isinstance(color, int) else (0, 0, 0)
        self.verbose = verbose

    @staticmethod
    def load_json(json_file):
        json_file = coerce_to_path_and_check_exist(json_file)
        with open(json_file, mode='r') as f:
            result = json.load(f)
        return result

    def run(self):
        for _, annot in self.annotations.items():
            img = self.convert(annot)
            if img is not None:
                img.save(self.output_dir / SEG_GROUND_TRUTH_FMT.format(annot['filename'].split('.')[0], self.out_ext))

    def convert(self, annot):
        name = annot['filename']
        if self.verbose:
            print_info('Converting VIA annotations for {}'.format(name))
        if not (self.input_dir / name).exists:
            print_error('Original image {} not found'.format(name))
            return None

        size = Image.open(self.input_dir / name).size
        img = Image.new(self.mode, size, color=self.background_color)
        draw = ImageDraw.Draw(img)

        for region in annot['regions']:
            shape = region['shape_attributes']
            if shape['name'] == 'circle':
                cx, cy, r = shape['cx'], shape['cy'], shape['r']
                bbox = [(cx - r, cy - r), (cx + r, cy + r)]
                draw.ellipse(bbox, fill=self.color)
            elif shape['name'] == 'ellipse':
                cx, cy, rx, ry = shape['cx'], shape['cy'], shape['rx'], shape['ry']
                bbox = [(cx - rx, cy - ry), (cx + rx, cy + ry)]
                draw.ellipse(bbox, fill=self.color)
            elif shape['name'] == 'polygon':
                polygon = list(zip(shape['all_points_x'], shape['all_points_y']))
                draw.polygon(polygon, fill=self.color)
            else:
                raise NotImplementedError('shape "{}" not implemented'.format(shape['name']))

        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VIA annotated regions into image files')
    parser.add_argument('-i', '--input_dir', nargs='?', type=str, required=True, help='Input directory where the'
                        'annotated images are, necessary to recover image sizes')
    parser.add_argument('-o', '--output_dir', nargs='?', type=str, required=True, help='Output directory')
    parser.add_argument('-f', '--file', nargs='?', type=str, required=True, help='Json file containing via annotations')
    args = parser.parse_args()

    conv = ViaJson2Image(args.input_dir, args.output_dir, json_file=args.file)
    conv.run()
