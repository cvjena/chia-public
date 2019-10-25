import uuid
import os
import csv
from PIL import Image

from chia.framework import configuration
from chia import knowledge
from chia.data import sample

_namespace_uuid = uuid.UUID("01a51cff-15ed-40cd-b921-bd4232d7288e")


class LNDWDataset:
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.base_path = configuration.get(
                "base_path", "/home/brust/datasets/lndw/dataset"
            )
        self.classes = []
        with open(os.path.join(self.base_path, "classes.csv")) as classes_file:
            reader = csv.reader(classes_file, delimiter=";")
            header = next(reader)
            fields = {
                "folder": "ID",
                "class_name": "Class Name",
                "individual_id": "No.",
                "grade": "Grade",
            }
            fields = {k: header.index(v) for k, v in fields.items()}
            for line in reader:
                self.classes += [{k: line[v] for k, v in fields.items()}]

            self.viable_classes = [
                class_ for class_ in self.classes if self._viable(class_)
            ]

    def _viable(self, class_):
        return float(class_["grade"]) <= 3.0

    def _filenames(self):
        return [
            "01W.jpg",
            "02SW.jpg",
            "03S.jpg",
            "04SE.jpg",
            "05E.jpg",
            "06NE.jpg",
            "07N.jpg",
            "08NW.jpg",
            "09TOP.jpg",
        ]

    def _build_sample(self, class_, filename, label_resource_id, individuals):
        # Open and resize image
        the_image = Image.open(
            os.path.join(self.base_path, f"{int(class_['folder']):02d}", filename)
        )
        the_image = the_image.resize((384, 384), Image.ANTIALIAS)

        if individuals:
            label_string = f"{class_['class_name']}{int(class_['individual_id']):02d}"
        else:
            label_string = f"{class_['class_name']}"

        # Build sample
        the_sample = (
            sample.Sample(
                source=self.__class__.__name__,
                uid=uuid.uuid5(_namespace_uuid, f"{class_}.{filename}"),
            )
            .add_resource(self.__class__.__name__, "input_img_np", the_image)
            .add_resource(self.__class__.__name__, label_resource_id, label_string)
        )
        return the_sample

    def get_train_pool(self, label_resource_id, individuals=False):
        samples = []
        for class_ in self.viable_classes:
            for filename in self._filenames()[:8]:
                the_sample = self._build_sample(
                    class_, filename, label_resource_id, individuals
                )
                samples += [the_sample]

        return samples

    def get_test_pool(self, label_resource_id, individuals=False):
        samples = []
        for class_ in self.viable_classes:
            for filename in self._filenames()[8:]:
                the_sample = self._build_sample(
                    class_, filename, label_resource_id, individuals
                )
                samples += [the_sample]

        return samples

    def get_kb(self):
        return knowledge.KnowledgeBase()
