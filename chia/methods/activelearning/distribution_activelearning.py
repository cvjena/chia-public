import random
from abc import ABC, abstractmethod
from typing import Hashable, Sequence, Tuple

from chia.methods.activelearning import ActiveLearningMethod


class OutputDistributionActiveLearningMethod(ActiveLearningMethod, ABC):
    @abstractmethod
    def distribution_score(self, distribution: Sequence[Tuple[Hashable, float]]):
        pass

    def score(
        self,
        samples,
        score_resource_id,
        prediction_dist_resource_id="_al_label_prediction_dist",
        **kwargs
    ):
        if not all(
            [sample.has_resource(prediction_dist_resource_id) for sample in samples]
        ):
            samples_ = self.model.predict_probabilities(
                samples, prediction_dist_resource_id
            )
        else:
            samples_ = samples
        samples_ = [
            sample.add_resource(
                self.__class__.__name__,
                score_resource_id,
                self.distribution_score(
                    sample_.get_resource(prediction_dist_resource_id)
                ),
            )
            for sample, sample_ in zip(samples, samples_)
        ]
        return samples_


class OneVsTwoActiveLearningMethod(OutputDistributionActiveLearningMethod):
    def distribution_score(self, distribution):
        if len(distribution) >= 2:
            sorted_probabilities = sorted([p for c, p in distribution], reverse=True)
            onevstwo = 1 - (sorted_probabilities[0] - sorted_probabilities[1])
            return onevstwo
        else:
            # Fall back if there are less than two classes
            return random.uniform(0.0, 1.0)
