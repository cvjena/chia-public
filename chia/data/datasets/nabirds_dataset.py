import uuid
import os
import json
from PIL import Image
import numpy as np

from chia.framework import configuration
from chia.data import sample
from chia import knowledge


_namespace_uid = "NABirds"


class NABirdsDataset:
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.base_path = configuration.get("base_path", "/home/datasets/nabirds")
            self.target_size = configuration.get("target_size", (384, 384))

        with open(os.path.join(self.base_path, "classes.txt")) as cls:
            lines = [x.strip() for x in cls]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            tuples = [(int(k), str(v)) for (k, v) in tuples]

            self.nabirds_id_to_label = {k: f"NAB:{int(k):03d}{v}" for (k, v) in tuples}

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

                    _nabirds_ids_with_instances = {
                        id for (img, id, tt) in combinedtuples
                    }

                    self._nabirds_training_tuples = [
                        (img, id) for (img, id, tt) in combinedtuples if tt == 1
                    ]
                    self._nabirds_validation_tuples = [
                        (img, id) for (img, id, tt) in combinedtuples if tt == 0
                    ]

                    _nabirds_all_image_ids = [img for (img, id, tt) in combinedtuples]

        with open(os.path.join(self.base_path, "hierarchy.txt")) as hie:
            lines = [x.strip() for x in hie]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            self.tuples = [
                (self.nabirds_id_to_label[int(k)], self.nabirds_id_to_label[int(v)])
                for (k, v) in tuples
            ]

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
        return (
            sample.Sample(
                source=self.__class__.__name__,
                uid=f"{_namespace_uid}:{split}:{image_id}",
            )
            .add_resource(
                self.__class__.__name__,
                label_resource_id,
                self.nabirds_id_to_label[label_id],
            )
            .add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(self.base_path, "images", self.iid_dict[image_id]),
            )
            .add_lazy_resource(
                self.__class__.__name__, "input_img_np", self._load_from_location
            )
        )

    def _load_from_location(self, sample_):
        return np.asarray(
            Image.open(sample_.get_resource("image_location")).resize(
                self.target_size, Image.ANTIALIAS
            )
        )

    def get_hypernymy_relation_source(self):
        return knowledge.StaticRelationSource(self.tuples)
