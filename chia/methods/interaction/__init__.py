from abc import ABC, abstractmethod


class InteractionMethod(ABC):
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
