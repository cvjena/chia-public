import numpy as np
import pickle as pkl
import uuid
import os

from chia.data import datasets
from chia.data import sample
from chia import configuration

_namespace_uuid = uuid.UUID("280c9580-aaf4-4dab-9f0b-3c3b52e00bb8")


class CORe50Dataset:
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.base_path = configuration.get(
                "base_path", "/home/brust/datasets/core50"
            )

        self.imgs = np.load(
            os.path.join(self.base_path, "core50_imgs.npy"), mmap_mode="r"
        )

        with open(os.path.join(self.base_path, "paths.pkl"), "rb") as paths_file:
            self.paths = pkl.load(paths_file)

        with open(
            os.path.join(self.base_path, "labels2names.pkl"), "rb"
        ) as labels_to_names_file:
            self.labels_to_names = pkl.load(labels_to_names_file)

        self.path_to_index = {path: index for index, path in enumerate(self.paths)}

    def get_train_pools(self, scenario, run, label_resource_id):

        pass

    def get_pool_for(self, scenario, run, batch, label_resource_id):
        # Find the data
        scenario = str(scenario).lower()
        assert scenario in ["ni", "nc", "nic"]

        filelist_path = os.path.join(
            self.base_path,
            "batches_filelists",
            f"{scenario.upper()}_inc",
            f"run{run:d}",
            f"{batch}_filelist.txt",
        )

        # Find appropriate label map
        label_map = self.labels_to_names[scenario][run]

        samples = []

        with open(filelist_path) as filelist:
            for line in filelist:
                path, class_id = line.strip().split(" ")
                samples += [
                    sample.Sample(
                        source=self.__class__.__name__,
                        uid=uuid.uuid5(_namespace_uuid, str(path)),
                    )
                    .add_resource(
                        self.__class__.__name__,
                        "input_img_np",
                        self.imgs[self.path_to_index[path]],
                    )
                    .add_resource(
                        self.__class__.__name__,
                        label_resource_id,
                        label_map[int(class_id)],
                    )
                ]

        return samples
