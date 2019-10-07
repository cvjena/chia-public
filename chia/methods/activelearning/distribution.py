from abc import ABC, abstractmethod

from chia.methods.activelearning import ActiveLearningMethod
from chia.methods.incrementallearning import ProbabilityOutputModel


class OutputDistributionActiveLearningMethod(ActiveLearningMethod, ABC):
    def __init__(self, model: ProbabilityOutputModel):
        self.model = model

    @abstractmethod
    def distribution_score(self, distribution):
        pass

    def score(self, samples):
        # TODO check if necessary
        samples_ = self.model.predict_probabilities(samples)
        samples_ = [sample.add_resource(self.__class__.__name__, 'score', self.distribution_score(sample_.get_resource('label_prediction_dist'))) for sample, sample_ in zip(samples, samples_)]
        return samples_


class OneVsTwoActiveLearningMethod(OutputDistributionActiveLearningMethod):
    def distribution_score(self, distribution):
        assert len(distribution) >= 2
        sorted_probabilities = sorted([p for c, p in distribution], reverse=True)
        onevstwo = 1 - (sorted_probabilities[0] - sorted_probabilities[1])
        return onevstwo
