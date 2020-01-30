import os

import numpy as np
from PIL import Image

from chia import knowledge
from chia.data import datasets, sample
from chia.framework import configuration, robustness

_namespace_uid = "NABirds"


class NABirdsDataset(datasets.Dataset):
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.base_path = configuration.get_system("NABirdsDataset.base_path")
            self.side_length = configuration.get("side_length", 224)
            self.use_lazy_mode = configuration.get("use_lazy_mode", True)

        with open(os.path.join(self.base_path, "classes.txt")) as cls:
            lines = [x.strip() for x in cls]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            tuples = [(int(k), str(v)) for (k, v) in tuples]

            self.nabirds_id_to_label = {
                k: f"{_namespace_uid}::{int(k):03d}{v}" for (k, v) in tuples
            }

            self.nabirds_ids = {k for (k, v) in tuples}

            if len([k for (k, v) in tuples]) != len({k for (k, v) in tuples}):
                print("Non-unique IDs found!")
                quit(-1)

        with open(os.path.join(self.base_path, "image_class_labels.txt")) as lab:
            with open(os.path.join(self.base_path, "train_test_split.txt")) as tts:
                with open(os.path.join(self.base_path, "images.txt")) as iid:
                    lablines = [x.strip() for x in lab]
                    labtuples = [x.split(sep=" ", maxsplit=1) for x in lablines]
                    labtuples = [(str(k), int(v)) for (k, v) in labtuples]

                    ttslines = [x.strip() for x in tts]
                    ttstuples = [x.split(sep=" ", maxsplit=1) for x in ttslines]
                    ttstuples = [(str(k), int(v)) for (k, v) in ttstuples]

                    iidlines = [x.strip() for x in iid]
                    iidtuples = [x.split(sep=" ", maxsplit=1) for x in iidlines]
                    iidtuples = [(str(k), str(v)) for (k, v) in iidtuples]
                    self.iid_dict = {k: v for (k, v) in iidtuples}

                    combinedtuples = [a + b for (a, b) in zip(labtuples, ttstuples)]
                    mismatches = [a != c for (a, b, c, d) in combinedtuples]
                    if any(mismatches):
                        print("Mismatch between tts and label files!")
                        quit(-1)

                    combinedtuples = [(a, b, d) for (a, b, c, d) in combinedtuples]

                    self._nabirds_training_tuples = [
                        (img, id) for (img, id, tt) in combinedtuples if tt == 1
                    ]
                    self._nabirds_validation_tuples = [
                        (img, id) for (img, id, tt) in combinedtuples if tt == 0
                    ]

        with open(os.path.join(self.base_path, "hierarchy.txt")) as hie:
            lines = [x.strip() for x in hie]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            self.tuples = [
                (self.nabirds_id_to_label[int(k)], self.nabirds_id_to_label[int(v)])
                for (k, v) in tuples
            ]

    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return 1

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_train_pool(label_resource_id)

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_test_pool(label_resource_id)

    def namespace(self):
        return _namespace_uid

    def relations(self):
        return ["hypernymy"]

    def relation(self, key):
        if key == "hypernymy":
            return self.get_hypernymy_relation_source()
        else:
            raise ValueError(f'Unknown relation "{key}"')

    def get_train_pool(self, label_resource_id):
        return [
            self._build_sample(image, id, label_resource_id, "train")
            for image, id in self._nabirds_training_tuples
        ]

    def get_test_pool(self, label_resource_id):
        return [
            self._build_sample(image, id, label_resource_id, "test")
            for image, id in self._nabirds_validation_tuples
        ]

    def _build_sample(self, image_id, label_id, label_resource_id, split):
        sample_ = sample.Sample(
            source=self.__class__.__name__, uid=f"{_namespace_uid}::{split}:{image_id}"
        ).add_resource(
            self.__class__.__name__,
            label_resource_id,
            self.nabirds_id_to_label[label_id],
        )
        if self.use_lazy_mode:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(self.base_path, "images", self.iid_dict[image_id]),
            ).add_lazy_resource(
                self.__class__.__name__, "input_img_np", self._load_from_location
            )
        else:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(self.base_path, "images", self.iid_dict[image_id]),
            )
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "input_img_np",
                self._load_from_location(sample_),
            )
        return sample_

    def _load_from_location(self, sample_):
        im = robustness.NetworkResistantImage.open(
            sample_.get_resource("image_location")
        ).resize((self.side_length, self.side_length), Image.ANTIALIAS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im)

    def get_hypernymy_relation_source(self):
        return knowledge.StaticRelationSource(self.tuples)
