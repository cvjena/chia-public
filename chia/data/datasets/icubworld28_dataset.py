import os
import imageio

from chia import knowledge
from chia.data import sample, datasets
from chia.framework import configuration

_namespace_uid = "iCubWorld28"

_icubworld28_labels_to_wordnet = [
    ("cup", "cup.n.01"),
    ("dishwashing-detergent", "dishwasher_detergent.n.01"),
    ("laundry-detergent", "laundry_detergent.n.01"),
    ("plate", "plate.n.04"),
    ("soap", "bar_soap.n.01"),
    ("sponge", "sponge.n.01"),
    ("sprayer", "atomizer.n.01"),
]


class iCubWorld28Dataset(datasets.Dataset):
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.base_path = configuration.get(
                "base_path", "/home/brust/datasets/icubworld28"
            )

    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return 4

    def test_pool_count(self):
        return 4

    def train_pool(self, index, label_resource_id):
        return self.get_train_pool_for(index + 1, label_resource_id)

    def test_pool(self, index, label_resource_id):
        return self.get_test_pool_for(index + 1, label_resource_id)

    def namespace(self):
        return _namespace_uid

    def relations(self):
        return ["hypernymy"]

    def relation(self, key):
        if key == "hypernymy":
            return self.get_hypernymy_relation_source()
        else:
            raise ValueError(f'Unknown relation "{key}"')

    def get_train_pool_for(self, day, label_resource_id):
        return self._build_samples("train", day, label_resource_id)

    def get_test_pool_for(self, day, label_resource_id):
        return self._build_samples("test", day, label_resource_id)

    def _build_samples(self, split, day, label_resource_id):
        assert day > 0
        samples = []
        for (category, wordnet_synset) in _icubworld28_labels_to_wordnet:
            for individual in range(1, 4 + 1):
                sample_dir = os.path.join(
                    self.base_path,
                    split,
                    f"day{day}",
                    category,
                    f"{category}{individual}",
                )
                for filename in sorted(os.listdir(sample_dir)):
                    samples += [
                        sample.Sample(
                            source=self.__class__.__name__,
                            uid=f"{_namespace_uid}::{split}:{day}:"
                            + f"{category}{individual}:{filename}",
                        )
                        .add_resource(
                            self.__class__.__name__,
                            label_resource_id,
                            f"{_namespace_uid}::{category}{individual}",
                        )
                        .add_resource(
                            self.__class__.__name__,
                            "image_input_np",
                            imageio.imread(os.path.join(sample_dir, filename)),
                        )
                    ]
        return samples

    def get_hypernymy_relation_source(self):
        relation = []
        for l, s in _icubworld28_labels_to_wordnet:
            for individual in range(1, 4 + 1):
                relation += [(f"{_namespace_uid}::{l}{individual}", f"WordNet3.0::{s}")]
        return knowledge.StaticRelationSource(relation)
