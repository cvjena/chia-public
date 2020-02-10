import unittest

import numpy as np

from chia.framework import ioqueue

_ITEM_COUNT = 100


class IoQueueTestCase(unittest.TestCase):
    def test_all_implementations(self):
        implementations = ["threading", "multiprocessing", "synchronous"]
        for implementation in implementations:
            self.subTest(f"Testing implementation: {implementation}")

            test_data = [list(np.random.random(32)) for _ in range(_ITEM_COUNT)]

            def test_data_gen():
                for test_datum in test_data:
                    yield test_datum

            expected_result = [result for result in test_data_gen()]

            actual_result = [
                result
                for result in ioqueue.make_generator_faster(
                    test_data_gen, implementation
                )
            ]

            self.assertEqual(expected_result, actual_result, "These should be equal")
            return
