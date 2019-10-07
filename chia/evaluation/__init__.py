from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def update(self, pool):
        return None

    @abstractmethod
    def result(self):
        return None

    @abstractmethod
    def reset(self):
        pass


class MultiEvaluator(Evaluator):
    def __init__(self):
        self._children = []

    def add(self, evaluator):
        self._children.append(evaluator)

    def update(self, samples):
        for evaluator in self._children:
            evaluator.update(samples)

    def result(self):
        result_dict = dict()
        for evaluator in self._children:
            result_dict.update(evaluator.result())
        return result_dict

    def reset(self):
        for evaluator in self._children:
            evaluator.reset()


class AccuracyEvaluator(Evaluator):
    def __init__(self):
        self.correct_count = 0
        self.sample_count = 0

    def reset(self):
        self.correct_count = 0
        self.sample_count = 0

    def update(self, samples):
        for sample in iter(samples):
            self.sample_count += 1
            if sample.get_resource('label_gt') == sample.get_resource('label_prediction'):
                self.correct_count += 1

    def result(self):
        return {'accuracy': float(self.correct_count) / float(self.sample_count)}
