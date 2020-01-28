from networkx.algorithms import shortest_paths

from . import Evaluator


class HierarchicalEvaluator(Evaluator):
    def __init__(self, kb):
        self.kb = kb
        self.reset()

    def reset(self):
        self.running_distance = 0
        self.sample_count = 0
        self.sample_count_confused = 0

    def update(self, samples, gt_resource_id, prediction_resource_id):
        for sample in iter(samples):
            gt_uid = sample.get_resource(gt_resource_id)
            pred_uid = sample.get_resource(prediction_resource_id)
            ugraph = self.kb.all_relations["hypernymy"]["ugraph"]
            if gt_uid != pred_uid:
                self.running_distance += shortest_paths.shortest_path_length(
                    ugraph, gt_uid, pred_uid
                )
                self.sample_count_confused += 1
            self.sample_count += 1

    def result(self):
        return {
            "semantic_distance": float(self.running_distance)
            / float(self.sample_count),
            "semantic_distance_confused": float(self.running_distance)
            / float(self.sample_count_confused),
        }
