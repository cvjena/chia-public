import functools
import unittest

from chia.data import datasets


class DatasetTestCase(unittest.TestCase):
    def test_all_datasets(self):
        for dataset_key in datasets.datasets():
            dataset = datasets.dataset(dataset_key)
            with self.subTest("Testing dataset", dataset=dataset_key):
                for setup in dataset.setups():
                    with self.subTest("Testing setup", setup=setup):
                        dataset.setup(**setup)
                        self.run_on_dataset(dataset)

    def run_on_dataset(self, dataset):
        # Data
        train_pool_count = dataset.train_pool_count()
        test_pool_count = dataset.test_pool_count()
        self.assertGreater(train_pool_count, 0, "There are train pools")
        self.assertGreater(test_pool_count, 0, "There are test pools")
        self.assertIsNotNone(dataset.namespace())

        train_sample_uids = [set()] * train_pool_count
        test_sample_uids = [set()] * test_pool_count

        for train_pool_index in range(train_pool_count):
            train_pool = dataset.train_pool(train_pool_index, "label_gt")
            self.assertGreater(len(train_pool), 0, "Train pool is not empty")

            for sample in train_pool:
                self.run_on_sample(dataset, sample)
                train_sample_uids[train_pool_index] |= {sample.get_resource("uid")}

        for test_pool_index in range(test_pool_count):
            test_pool = dataset.test_pool(test_pool_index, "label_gt")
            self.assertGreater(len(test_pool), 0, "Test pool is not empty")

            for sample in test_pool:
                self.run_on_sample(dataset, sample)
                test_sample_uids[test_pool_index] |= {sample.get_resource("uid")}

        # Train / test intersection
        all_train_uids = functools.reduce(lambda x, y: x | y, train_sample_uids, set())
        all_test_uids = functools.reduce(lambda x, y: x | y, test_sample_uids, set())

        train_test_intersection = all_train_uids.intersection(all_test_uids)

        self.assertTrue(
            len(train_test_intersection) == 0, "Train and test set have no intersection"
        )

        # Relations
        self.assertIn(
            "hypernymy", dataset.relations(), "Dataset provides a hypernymy relation"
        )

    def run_on_sample(self, dataset, sample):
        self.assertTrue(
            str(sample.get_resource("uid")).startswith(f"{dataset.namespace()}::"),
            "Sample UID has correct namespace",
        )
        self.assertTrue(
            str(sample.get_resource("label_gt")).startswith(f"{dataset.namespace()}::"),
            "Sample label has correct namespace",
        )


def test_generator(dataset):
    # This pattern is from
    # https://stackoverflow.com/questions/32899/how-do-you-generate-dynamic-parameterized-unit-tests-in-python
    def test(self: DatasetTestCase):
        self.run_on_dataset(dataset)

    return test


if __name__ == "__main__":
    unittest.main()
