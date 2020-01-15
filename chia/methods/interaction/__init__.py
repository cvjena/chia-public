from abc import ABC, abstractmethod

from chia.methods.interaction import noisy_oracle_interaction  # noqa


class InteractionMethod(ABC):
    def __init__(self, kb=None):
        self._kb = kb

    @abstractmethod
    def query_annotations_for(self, samples, gt_resource_id, ann_resource_id):
        return None


class OracleInteractionMethod(InteractionMethod):
    def query_annotations_for(self, samples, gt_resource_id, ann_resource_id):
        return [
            sample.add_resource(
                self.__class__.__name__,
                ann_resource_id,
                sample.get_resource(gt_resource_id),
            )
            for sample in samples
        ]


_method_mapping = {
    "Oracle": OracleInteractionMethod,
    "NoisyOracle": noisy_oracle_interaction.NoisyOracleInteractionMethod,
}


def methods():
    return _method_mapping.keys()


def method(key, kb):
    return _method_mapping[key](kb)
