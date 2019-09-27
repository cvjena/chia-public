from abc import ABC, abstractmethod

from chia.data.pool import Pool
from chia.data.sample import Sample


class ActiveLearningMethod(ABC):
    """ Requests samples from a pool and assigns a score to each sample. """
    @abstractmethod
    def score(self, samples):
        raise NotImplementedError


class DummyActiveLearningMethod(ActiveLearningMethod):
    def score(self, samples):
        scores = [0.0] * len(samples)
        return zip(samples, scores)
