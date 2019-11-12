from abc import ABC, abstractmethod

from chia.data.pool import Pool
from chia.data.sample import Sample

import random


class ActiveLearningMethod(ABC):
    def __init__(self, model, kb):
        self.model = model
        self.kb = kb

    """ Requests samples from a pool and assigns a score to each sample. """

    @abstractmethod
    def score(self, samples, score_resource_id, **kwargs):
        raise NotImplementedError


class RandomActiveLearningMethod(ActiveLearningMethod):
    def score(self, samples, score_resource_id, **kwargs):
        return [
            sample.add_resource(
                self.__class__.__name__, score_resource_id, random.uniform(0.0, 1.0)
            )
            for sample in samples
        ]


from chia.methods.activelearning import distribution_activelearning

_method_mapping = {
    "Random": RandomActiveLearningMethod,
    "1vs2": distribution_activelearning.OneVsTwoActiveLearningMethod,
}


def methods():
    return _method_mapping.keys()


def method(key, model, kb):
    return _method_mapping[key](model, kb)
