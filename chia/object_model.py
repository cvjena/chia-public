from abc import ABC, abstractmethod

from .data import Pool, Sample


class SampleFilter(ABC):
    @abstractmethod
    def filter_(self, samples):
        return samples

