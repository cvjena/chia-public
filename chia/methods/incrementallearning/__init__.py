from abc import ABC, abstractmethod


class IncrementalModel(ABC):
    @abstractmethod
    def observe(self, samples, gt_resource_id, progress_callback=None):
        return None

    @abstractmethod
    def predict(self, samples, prediction_resource_id):
        return None

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass


class ProbabilityOutputModel(IncrementalModel, ABC):
    def predict(self, samples, prediction_resource_id):
        samples_ = self.predict_probabilities(samples, prediction_resource_id + "_dist")
        samples_ = [
            sample.add_resource(
                self.__class__.__name__,
                prediction_resource_id,
                sorted(
                    sample.get_resource(prediction_resource_id + "_dist"),
                    key=lambda x: x[1],
                    reverse=True,
                )[0][0],
            )
            for sample in samples_
        ]
        return samples_

    @abstractmethod
    def predict_probabilities(self, samples, prediction_dist_resource_id):
        pass


from chia.methods.incrementallearning import (  # noqa isort:skip
    keras_dfn,
    keras_fastsingleshot,
)

_method_mapping = {
    "keras::DFN": keras_dfn.DFNKerasIncrementalModel,
    "keras::FastSingleShot": keras_fastsingleshot.FastSingleShotKerasIncrementalModel,
}


def methods():
    return _method_mapping.keys()


def method(key, cls):
    return _method_mapping[key](cls)
