from abc import ABC, abstractmethod
import joblib

from chia.methods.incrementallearning import ProbabilityOutputModel
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


class SKLearnIncrementalModel(ProbabilityOutputModel, ABC):
    def __init__(self, kb, cls):
        self.kb = kb
        self.X = []
        self.y = []
        self.cls = cls

    def observe(self, samples, gt_resource_id):
        for sample in samples:
            self.X.append(sample.get_resource("input_img_np").flatten())
            self.y.append(sample.get_resource(gt_resource_id))

        self.cls.fit(self.X, self.y)

    def save(self, path):
        joblib.dump(self.cls, path)

    def restore(self, path):
        joblib.load(self.cls, path)


class SKLearnSVCIncrementalModel(SKLearnIncrementalModel, ProbabilityOutputModel):
    def __init__(self, kb):
        SKLearnIncrementalModel.__init__(
            self, kb, SVC(kernel="poly", C=10, gamma="scale", probability=True)
        )

    def predict_probabilities(self, samples, prediction_dist_resource_id):
        return [
            sample.add_resource(
                self.__class__.__name__,
                prediction_dist_resource_id,
                list(
                    zip(
                        self.cls.classes_,
                        self.cls.predict_proba(
                            [sample.get_resource("input_img_np").flatten()]
                        )[0],
                    )
                ),
            )
            for sample in samples
        ]


class SKLearnKNNIncrementalModel(SKLearnIncrementalModel, ProbabilityOutputModel):
    def __init__(self, kb):
        SKLearnIncrementalModel.__init__(self, kb, KNeighborsClassifier())

    def predict_probabilities(self, samples, prediction_dist_resource_id):
        return [
            sample.add_resource(
                self.__class__.__name__,
                prediction_dist_resource_id,
                list(
                    zip(
                        self.cls.classes_,
                        self.cls.predict_proba(
                            [sample.get_resource("input_img_np").flatten()]
                        )[0],
                    )
                ),
            )
            for sample in samples
        ]


class SKLearnMLPIncrementalModel(SKLearnIncrementalModel):
    def __init__(self, kb):
        SKLearnIncrementalModel.__init__(self, kb, MLPClassifier())

    def predict_probabilities(self, samples, prediction_dist_resource_id):
        return [
            sample.add_resource(
                self.__class__.__name__,
                prediction_dist_resource_id,
                list(
                    zip(
                        self.cls.classes_,
                        self.cls.predict_proba(
                            [sample.get_resource("input_img_np").flatten()]
                        )[0],
                    )
                ),
            )
            for sample in samples
        ]
