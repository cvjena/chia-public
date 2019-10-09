from abc import ABC, abstractmethod


class IncrementalModel(ABC):
    @abstractmethod
    def observe(self, samples):
        return None

    @abstractmethod
    def predict(self, samples):
        return None

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass


class ProbabilityOutputModel(IncrementalModel, ABC):
    def predict(self, samples):
        samples_ = self.predict_probabilities(samples)
        samples_ = [sample.add_resource(self.__class__.__name__, 'label_prediction', sorted(sample.get_resource('label_prediction_dist'), key=lambda x: x[1], reverse=True)[0][0]) for sample in samples_]
        return samples_

    @abstractmethod
    def predict_probabilities(self, samples):
        pass


class DummyIncrementalModel(IncrementalModel):
    def observe(self, samples):
        return None

    def predict(self, samples):
        return [sample.add_resource(self.__class__.__name__, 'label_prediction', sample.get_resource('label_gt')) for sample in samples]

    def save(self, path):
        raise NotImplementedError

    def restore(self, path):
        raise NotImplementedError

