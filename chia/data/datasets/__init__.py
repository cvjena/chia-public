import uuid

from chia.data.pool import FixedPool
from chia.data.sample import Sample
from chia.knowledge import KnowledgeBase


_namespace_uuid = uuid.UUID("ef3158cf-056a-4ace-a766-4ea93ace5e59")


class FashionMNISTDataset:
    def __init__(self):
        import tensorflow as tf

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (
            self.test_images,
            self.test_labels,
        ) = fashion_mnist.load_data()

    def get_train_pool(self, label_resource_id):
        return FashionMNISTDataset._get_pool(
            self.train_images, self.train_labels, "train", label_resource_id
        )

    def get_test_pool(self, label_resource_id):
        return FashionMNISTDataset._get_pool(
            self.test_images, self.test_labels, "test", label_resource_id
        )

    def get_kb(self):
        return KnowledgeBase()

    @staticmethod
    def _get_pool(images, labels, uid_suffix, label_resource_id):
        assert images.shape[0] == labels.shape[0]

        samples = []
        for i in range(images.shape[0]):
            samples += [
                Sample(
                    source="FashionMNISTDataset",
                    uid=uuid.uuid5(_namespace_uuid, f"{i:6d}.{uid_suffix}"),
                )
                .add_resource(
                    "FashionMNISTDataset",
                    "input_img_np",
                    images[i].reshape([28, 28, 1]).repeat(3, axis=2),
                )
                .add_resource("FashionMNISTDataset", label_resource_id, labels[i])
            ]

        return FixedPool(samples)
