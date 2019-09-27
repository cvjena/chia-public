from abc import ABC, abstractmethod


class Pool(ABC):
    @abstractmethod
    def request(self, sample_count: int, remove: bool):
        raise NotImplementedError()


class FixedPool(Pool):
    def __init__(self, samples):
        self.samples = samples
        self.pos = 0

    def request(self, sample_count: int, remove: bool):
        return None