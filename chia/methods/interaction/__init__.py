from abc import ABC, abstractmethod


class InteractionMethod(ABC):
    @abstractmethod
    def annotate(self, samples):
        return None
