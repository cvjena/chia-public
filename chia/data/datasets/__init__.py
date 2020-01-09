import abc


class Dataset(abc.ABC):
    @abc.abstractmethod
    def setup(self, **kwargs):
        pass

    def setups(self):
        return [dict()]

    @abc.abstractmethod
    def train_pool_count(self):
        pass

    @abc.abstractmethod
    def test_pool_count(self):
        pass

    @abc.abstractmethod
    def train_pool(self, index, label_resource_id):
        pass

    @abc.abstractmethod
    def test_pool(self, index, label_resource_id):
        pass

    @abc.abstractmethod
    def namespace(self):
        pass

    @abc.abstractmethod
    def relations(self):
        pass

    @abc.abstractmethod
    def relation(self, key):
        pass


from chia.data.datasets import (
    core50_dataset,
    icifar_dataset,
    icubworld28_dataset,
    ilsvrc2012_dataset,
    lndw_dataset,
    nabirds_dataset,
)  # noqa


_dataset_mapping = {
    "CORe50": core50_dataset.CORe50Dataset,
    "iCIFAR": icifar_dataset.iCIFARDataset,
    "iCubWorld28": icubworld28_dataset.iCubWorld28Dataset,
    "ILSVRC2012": ilsvrc2012_dataset.ILSVRC2012Dataset,
    "LNdW": lndw_dataset.LNDWDataset,
    "NABirds": nabirds_dataset.NABirdsDataset,
}


def datasets():
    return _dataset_mapping.keys()


def dataset(key) -> Dataset:
    return _dataset_mapping[key]()
