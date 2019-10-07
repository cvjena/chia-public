from abc import ABC, abstractmethod
from collections import MutableSequence


class Pool(MutableSequence, ABC):
    def remove_multiple(self, objects):
        for object_ in objects:
            self.remove(object_)


class FixedPool(list, Pool):
    def __init__(self, samples=None):
        if samples is not None:
            list.__init__(self, samples)
        else:
            list.__init__(self)
