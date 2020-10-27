import sys
sys.path.insert(1, '/workspace/Deep SVDD/')
from datasets.baemin import baemin_Dataset
from datasets.ADbaemin import ADbaemin_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('baemin', 'ADbaemin')
    assert dataset_name in implemented_datasets

    dataset = None
    
    if dataset_name == 'baemin':
        dataset = baemin_Dataset(root=data_path, normal_class=normal_class)
        
    if dataset_name == 'ADbaemin':
        dataset = ADbaemin_Dataset(root=data_path, normal_class=normal_class)

    return dataset
