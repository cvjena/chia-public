import tensorflow as tf
import uuid
import random
import numpy as np

from chia.data import sample

_namespace_uuid = uuid.UUID("458e91ef-6c48-428b-9b9e-a136d1d8fa7f")
_label_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


class iCIFARDataset:
    def __init__(self):
        (self.train_X, self.train_y), (
            self.test_X,
            self.test_y,
        ) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

        self.sequence_seed = 19219
        self._update_sequence()

        sorting_sequence_train = np.argsort(self.train_y[:, 0], kind="stable")
        self.train_X = self.train_X[sorting_sequence_train]
        self.train_y = self.train_y[sorting_sequence_train]

    def _update_sequence(self):
        random.seed(self.sequence_seed)
        self.sequence = random.sample(range(100), 100)

    def get_train_pool_count(self, classes_per_batch):
        return 100 // classes_per_batch

    def get_train_pool_for(self, batch, label_resource_id, classes_per_batch):
        assert (100 % classes_per_batch) == 0
        batch_count = 100 // classes_per_batch
        assert batch < batch_count

        samples = []
        classes_for_pool = self.sequence[batch : batch + classes_per_batch]

        print(f"Retrieving images for classes {classes_for_pool}")
        for class_ in classes_for_pool:
            training_data_range = list(range(500 * class_, 500 * (class_ + 1)))
            class_X = self.train_X[500 * class_ : 500 * (class_ + 1)]
            class_y = self.train_y[500 * class_ : 500 * (class_ + 1)]
            samples += self._build_samples(
                class_X, class_y, training_data_range, label_resource_id, "train"
            )

        return samples

    def get_test_pool(self, label_resource_id):
        return self._build_samples(
            self.test_X, self.test_y, range(0, 10000), label_resource_id, "test"
        )

    def _build_samples(self, X, y, data_range, label_resource_id, prefix):
        assert X.shape[0] == len(data_range)
        samples = []
        for i, data_id in enumerate(data_range):
            class_label = y[i, 0]
            np_image = X[i]
            samples += [
                sample.Sample(
                    source=self.__class__.__name__,
                    uid=uuid.uuid5(_namespace_uuid, f"{prefix}.{data_id}"),
                )
                .add_resource(self.__class__.__name__, "input_img_np", np_image)
                .add_resource(
                    self.__class__.__name__,
                    label_resource_id,
                    _label_names[int(class_label)],
                )
            ]
        return samples
