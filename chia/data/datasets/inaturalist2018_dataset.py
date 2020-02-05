import json
import os

import numpy as np
from PIL import Image

from chia import knowledge
from chia.data import datasets, sample
from chia.framework import configuration, robustness

_namespace_uid = "iNaturalist2018"


class iNaturalist2018Dataset(datasets.Dataset):
    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return 2

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        assert index == 0
        return self.pool_from_json("train2018.json", label_resource_id)

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.pool_from_json("val2018.json", label_resource_id)

    def namespace(self):
        return _namespace_uid

    def relations(self):
        return ["hypernymy"]

    def relation(self, key):
        if key == "hypernymy":
            return self.get_hypernymy_relation_source()
        else:
            raise ValueError(f'Unknown relation "{key}"')

    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.base_path = configuration.get_system(
                "iNaturalist2018Dataset.base_path"
            )
            self.side_length = configuration.get("side_length", 224)

        self._id_to_class = {}
        with open(os.path.join(self.base_path, "categories.json")) as json_file:
            json_data = json.load(json_file)
            for json_datum in json_data:
                self._id_to_class[json_datum["id"]] = json_datum["name"]

    def pool_from_json(self, filename, label_resource_id):
        with open(os.path.join(self.base_path, filename)) as json_file:
            json_data = json.load(json_file)
        image_list = json_data["images"]
        annotation_list = json_data["annotations"]
        annotations = {ann["image_id"]: ann["category_id"] for ann in annotation_list}

        return [
            self.build_sample(image_dict, label_resource_id, annotations)
            for image_dict in image_list
        ]

    def build_sample(self, image_dict, label_resource_id, annotations):
        image_filename = image_dict["file_name"]
        image_id = image_dict["id"]
        sample_ = sample.Sample(
            source=self.__class__.__name__, uid=f"{_namespace_uid}::{image_filename}"
        )
        sample_ = sample_.add_resource(
            self.__class__.__name__,
            label_resource_id,
            self._id_to_class[annotations[image_id]],
        )
        sample_ = sample_.add_resource(
            self.__class__.__name__,
            "image_location",
            os.path.join(self.base_path, image_filename),
        ).add_lazy_resource(
            self.__class__.__name__, "input_img_np", self._load_from_location
        )

        return sample_

    def get_hypernymy_relation_source(self):
        return knowledge.StaticRelationSource([])

    def _load_from_location(self, sample_):
        im = robustness.NetworkResistantImage.open(
            sample_.get_resource("image_location")
        ).resize((self.side_length, self.side_length), Image.ANTIALIAS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im)
