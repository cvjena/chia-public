import numpy as np
import pickle as pkl
import uuid
import os
import glob

from chia.data import sample
from chia.framework import configuration
from chia import knowledge

_namespace_uuid = uuid.UUID("280c9580-aaf4-4dab-9f0b-3c3b52e00bb8")
_labels_to_wordnet = [
    ("plug_adapter1", "plug.n.05"),
    ("plug_adapter2", "plug.n.05"),
    ("plug_adapter3", "plug.n.05"),
    ("plug_adapter4", "plug.n.05"),
    ("plug_adapter5", "plug.n.05"),
    ("mobile_phone1", "cellular_telephone.n.01"),
    ("mobile_phone2", "cellular_telephone.n.01"),
    ("mobile_phone3", "cellular_telephone.n.01"),
    ("mobile_phone4", "cellular_telephone.n.01"),
    ("mobile_phone5", "cellular_telephone.n.01"),
    ("scissor1", "scissors.n.01"),
    ("scissor2", "scissors.n.01"),
    ("scissor3", "scissors.n.01"),
    ("scissor4", "scissors.n.01"),
    ("scissor5", "scissors.n.01"),
    ("light_bulb1", "light_bulb.n.01"),
    ("light_bulb2", "light_bulb.n.01"),
    ("light_bulb3", "light_bulb.n.01"),
    ("light_bulb4", "light_bulb.n.01"),
    ("light_bulb5", "light_bulb.n.01"),
    ("can1", "soda_can.n.01"),
    ("can2", "soda_can.n.01"),
    ("can3", "soda_can.n.01"),
    ("can4", "soda_can.n.01"),
    ("can5", "soda_can.n.01"),
    ("glass1", "spectacles.n.01"),
    ("glass2", "sunglasses.n.01"),
    ("glass3", "sunglasses.n.01"),
    ("glass4", "sunglasses.n.01"),
    ("glass5", "sunglasses.n.01"),
    ("ball1", "ball.n.01"),
    ("ball2", "tennis_ball.n.01"),
    ("ball3", "football.n.02"),
    ("ball4", "ball.n.01"),
    ("ball5", "football.n.02"),
    ("marker1", "highlighter.n.02"),
    ("marker2", "highlighter.n.02"),
    ("marker3", "highlighter.n.02"),
    ("marker4", "highlighter.n.02"),
    ("marker5", "highlighter.n.02"),
    ("cup1", "cup.n.01"),
    ("cup2", "cup.n.01"),
    ("cup3", "cup.n.01"),
    ("cup4", "cup.n.01"),
    ("cup5", "cup.n.01"),
    ("remote_control1", "remote_control.n.01"),
    ("remote_control2", "remote_control.n.01"),
    ("remote_control3", "remote_control.n.01"),
    ("remote_control4", "remote_control.n.01"),
    ("remote_control5", "remote_control.n.01"),
]


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

    def get_train_pool_count(self, scenario, run):
        scenario = str(scenario).lower()
        assert scenario in ["ni", "nc", "nic"]
        filelist_filter = os.path.join(
            self.base_path,
            "batches_filelists",
            f"{scenario.upper()}_inc",
            f"run{run:d}",
            "train_batch_*_filelist.txt",
        )

        return len(glob.glob(filelist_filter))

    def get_run_count(self, scenario):
        scenario = str(scenario).lower()
        assert scenario in ["ni", "nc", "nic"]
        filelist_filter = os.path.join(
            self.base_path, "batches_filelists", f"{scenario.upper()}_inc", f"run*"
        )

        return len(glob.glob(filelist_filter))

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

    def get_hypernymy_relation_source(self):
        return knowledge.StaticRelationSource(
            [(l, f"WN:{s}") for l, s in _labels_to_wordnet]
        )
