import argparse

from synthetic.line import SyntheticLine

from utils import coerce_to_path_and_create_dir
from utils.logger import print_info
from utils.path import DATASETS_PATH, SYNTHETIC_LINE_DATASET_PATH


class SyntheticLineDatasetGenerator:
    """Create a dataset (train/val/test) by generating synthetic random lines."""

    def __init__(self, output_dir=SYNTHETIC_LINE_DATASET_PATH, verbose=True):
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.verbose = verbose
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

    def run(self, nb_train, nb_val=0.1, nb_test=0.1):
        if 0 < nb_val < 1:
            nb_val = int(nb_train * nb_val)
        if 0 < nb_test < 1:
            nb_test = int(nb_train * nb_test)
        shift = 0
        max_len_id = len(str(nb_train + nb_val + nb_test - 1))
        for name, nb in zip(['train', 'val', 'test'], [nb_train, nb_val, nb_test]):
            if self.verbose:
                print_info('Creating {} set...'.format(name))
            for k in range(shift, shift + nb):
                if self.verbose:
                    print_info('  Generating random lines with seed {}...'.format(k))
                d = SyntheticLine(seed=k)
                d.save('{}'.format(k).zfill(max_len_id), self.output_dir / name)
            shift += nb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a synthetic line dataset, with train, val, test splits')
    parser.add_argument('-d', '--dataset_name', nargs='?', type=str, help='Output dataset name', required=True)
    parser.add_argument('-n', '--nb_train', type=int, default=1000, help='Number of train samples to generate')
    args = parser.parse_args()

    output_dir = DATASETS_PATH / args.dataset_name

    gen = SyntheticLineDatasetGenerator(output_dir)
    gen.run(args.nb_train)
