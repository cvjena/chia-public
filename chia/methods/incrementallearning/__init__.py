from abc import ABC, abstractmethod


class IncrementalModel(ABC):
    @abstractmethod
    def observe(self, samples):
        return None
