from abc import ABC, abstractmethod

from chia.data.pool import Pool
from chia.data.sample import Sample

import random


class ActiveLearningMethod(ABC):
    """ Requests samples from a pool and assigns a score to each sample. """

    @abstractmethod
    def score(self, samples, score_resource_id):
        raise NotImplementedError


class RandomActiveLearningMethod(ActiveLearningMethod):
    def score(self, samples, score_resource_id):
        return [
            sample.add_resource(
                self.__class__.__name__, "score_resource_id", random.uniform(0.0, 1.0)
            )
            for sample in samples
        ]
