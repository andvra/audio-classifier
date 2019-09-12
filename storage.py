import h5py
from typing import List

class HDF5Dataset:
    def __init__(self, group_name: str, dataset_name: str, shape, dtype: str):
        self.__dataset_name = dataset_name
        self.__group_name = group_name
        self.__shape = shape
        self.__dtype = dtype

    def shape(self):
        return self.__shape

    def group_name(self):
        return self.__group_name

    def dataset_name(self):
        return self.__dataset_name

    def dtype(self):
        return self.__dtype

    def __str__(self):
        return f'Group: {self.group_name()} Dataset: {self.dataset_name()}'

HDF5Datasets = List[HDF5Dataset]

def create_hdf5(path: str, groups: HDF5Datasets) -> None:
    with h5py.File(path, "w") as f:
        unique_group_names = list(set(map(lambda x: x.group_name(), groups)))
        for group_name in unique_group_names:
            grp = f.create_group(group_name)
            filtered_datasets = list(filter(lambda x: x.group_name()==group_name, groups))
            for ds in filtered_datasets:
                grp.create_dataset(ds.dataset_name(), ds.shape(), dtype=ds.dtype())