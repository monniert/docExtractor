import argparse

from numpy.random import choice

from synthetic.document import SyntheticDocument

from utils import coerce_to_path_and_create_dir, use_seed
from utils.logger import print_info
from utils.path import DATASETS_PATH, SYNTHETIC_DOCUMENT_DATASET_PATH

RANDOM_DOC_HEIGHT_FREQ = 0.5


class SyntheticDocumentDatasetGenerator:
    """Create a dataset (train/val/test) by generating synthetic random documents."""

    def __init__(self, output_dir=SYNTHETIC_DOCUMENT_DATASET_PATH, merged_labels=True, baseline_as_label=False,
                 verbose=True):
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.merged_labels = merged_labels
        self.baseline_as_label = baseline_as_label
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
        kwargs = {'baseline_as_label': self.baseline_as_label, 'merged_labels': self.merged_labels}
        for name, nb in zip(['train', 'val', 'test'], [nb_train, nb_val, nb_test]):
            if self.verbose:
                print_info('Creating {} set...'.format(name))
            for k in range(shift, shift + nb):
                if self.verbose:
                    print_info('  Generating random document with seed {}...'.format(k))
                with use_seed(k):
                    random_height = choice([True, False], p=[RANDOM_DOC_HEIGHT_FREQ, 1 - RANDOM_DOC_HEIGHT_FREQ])
                    if random_height:
                        kwargs['height'] = None
                kwargs['seed'] = k
                d = SyntheticDocument(**kwargs)
                d.save('{}'.format(k).zfill(max_len_id), self.output_dir / name)
            shift += nb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a synthetic document dataset, with train, val, test splits')
    parser.add_argument('-d', '--dataset_name', nargs='?', type=str, help='Output dataset name', required=True)
    parser.add_argument('-n', '--nb_train', type=int, default=1000, help='Number of train samples to generate')
    parser.add_argument('-m', '--merged_labels', action='store_true', help='Merge labels into illustration and text')
    parser.add_argument('-b', '--baseline_as_label', action='store_true', help='Draw baseline labels instead of text'
                        'line ones')
    args = parser.parse_args()

    gen = SyntheticDocumentDatasetGenerator(DATASETS_PATH / args.dataset_name, merged_labels=args.merged_labels,
                                            baseline_as_label=args.baseline_as_label)
    gen.run(args.nb_train)
