import random

from chia.framework import configuration
from chia.methods.activelearning import distribution_activelearning

from chia.methods import semanticmeasures


class SemanticOneVsTwoActiveLearningMethod(
    distribution_activelearning.OutputDistributionActiveLearningMethod
):
    def __init__(self, model, kb):
        super().__init__(model, kb)
        with configuration.ConfigurationContext("SemanticOneVsTwoActiveLearningMethod"):
            self.semantic_measure_name = configuration.get(
                "semantic_measure", "Rada1989"
            )
            self.semantic_measure = semanticmeasures.method(
                self.semantic_measure_name, kb
            )

    def distribution_score(self, distribution):
        if len(distribution) >= 2:
            sorted_probabilities = sorted(
                distribution, key=lambda x: x[1], reverse=True
            )
            onevstwo = 1 - (sorted_probabilities[0][1] - sorted_probabilities[1][1])

            uida = sorted_probabilities[0][0]
            uidb = sorted_probabilities[1][0]
            return onevstwo * self.semantic_measure.measure(uida, uidb)
        else:
            # Fall back if there are less than two classes
            return random.uniform(0.0, 1.0)
