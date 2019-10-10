from abc import ABC
from collections import Sequence


class Pool(Sequence, ABC):
    def remove_multiple(self, samples):
        samples_to_remove = set(samples)
        return self.__class__(
            samples=[sample for sample in self if sample not in samples_to_remove]
        )


class FixedPool(list, Pool):
    def __init__(self, samples=None):
        if samples is not None:
            list.__init__(self, samples)
        else:
            list.__init__(self)
