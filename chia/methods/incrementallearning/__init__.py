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

