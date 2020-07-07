from .segmentation import AbstractSegDataset


def get_dataset(dataset_name):
    class Dataset(AbstractSegDataset):
        name = dataset_name

    return Dataset
