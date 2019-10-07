from abc import ABC, abstractmethod

from chia.data.pool import Pool
from chia.data.sample import Sample

import random


class ActiveLearningMethod(ABC):
    """ Requests samples from a pool and assigns a score to each sample. """
    @abstractmethod
    def score(self, samples):
        raise NotImplementedError


class DummyActiveLearningMethod(ActiveLearningMethod):
    def score(self, samples):
        return [sample.add_resource(self.__class__.__name__, 'score', random.uniform(0.0, 1.0)) for sample in samples]
