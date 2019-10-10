from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Hashable

from chia.methods.activelearning import ActiveLearningMethod
from chia.methods.incrementallearning import ProbabilityOutputModel


class OutputDistributionActiveLearningMethod(ActiveLearningMethod, ABC):
    def __init__(self, model: ProbabilityOutputModel):
        self.model = model

    @abstractmethod
    def distribution_score(self, distribution: Sequence[Tuple[Hashable, float]]):
        pass

    def score(self, samples, score_resource_id):
        # TODO check if predictions are necessary
        temp_prediction_dist_resource_id = "_al_label_prediction_dist"
        samples_ = self.model.predict_probabilities(
            samples, temp_prediction_dist_resource_id
        )
        samples_ = [
            sample.add_resource(
                self.__class__.__name__,
                score_resource_id,
                self.distribution_score(
                    sample_.get_resource(temp_prediction_dist_resource_id)
                ),
            )
            for sample, sample_ in zip(samples, samples_)
        ]
        return samples_


class OneVsTwoActiveLearningMethod(OutputDistributionActiveLearningMethod):
    def distribution_score(self, distribution):
        assert len(distribution) >= 2
        sorted_probabilities = sorted([p for c, p in distribution], reverse=True)
        onevstwo = 1 - (sorted_probabilities[0] - sorted_probabilities[1])
        return onevstwo
