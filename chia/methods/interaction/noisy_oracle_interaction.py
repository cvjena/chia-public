from typing import Optional

import networkx as nx
import numpy as np

from chia import knowledge
from chia.framework import configuration
from chia.methods import interaction


class NoisyOracleInteractionMethod(interaction.InteractionMethod):
    def __init__(self, kb):
        super().__init__(kb)
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.noise_model = configuration.get("noise_model", no_default=True)

            if self.noise_model == "Deng2014":
                self.relabel_fraction: float = configuration.get(
                    "relabel_fraction", no_default=True
                )
            elif self.noise_model == "Poisson":
                self.lambda_: float = configuration.get("lambda", no_default=True)
            elif self.noise_model == "Geometric":
                self.q: float = configuration.get("q", no_default=True)
            else:
                raise ValueError(f"Unknown noise model: {self.noise_model}")

            self.filter_imprecise = configuration.get("filter_imprecise", False)
            self.project_to_random_leaf = configuration.get(
                "project_to_random_leaf", False
            )

        self.last_concept_stamp = -1
        self.graph: Optional[nx.DiGraph] = None
        self.root = None

    def _apply_deng_noise(self, uid):
        if np.random.binomial(1, self.relabel_fraction):
            chosen_predecessor = next(
                self.graph.predecessors(uid)
            )  # TODO what to do if there is more than 1 parent?
            return chosen_predecessor
        else:
            return uid

    def _apply_geometric_noise(self, uid):
        target = np.random.geometric(1 - self.q) - 1
        return self._reduce_depth_to(uid, target)

    def _apply_poisson_noise(self, uid):
        target = np.random.poisson(self.lambda_)
        return self._reduce_depth_to(uid, target)

    def _reduce_depth_to(self, uid, depth_target):
        path_to_label = nx.shortest_path(self.graph, self.root, uid)
        final_depth = max(0, min(len(path_to_label) - 1, depth_target))
        return path_to_label[final_depth]

    def _project_to_random_leaf(self, uid):
        if self.graph.out_degree(uid) == 0:  # noqa
            return uid
        else:
            # List all descendants
            all_descendants = nx.descendants(self.graph, uid)

            # Use only leaves
            valid_descendants = list(
                filter(lambda n: self.graph.out_degree(n) == 0, all_descendants)  # noqa
            )

            return np.random.choice(valid_descendants)

    def _maybe_update_graphs(self):
        kb: knowledge.KnowledgeBase = self._kb
        if kb.get_concept_stamp() != self.last_concept_stamp:
            graph: nx.DiGraph = kb.all_relations["hypernymy"]["graph"]
            self.graph = graph.reverse(copy=True)
            self.root = next(nx.topological_sort(self.graph))
            print(self.graph.nodes)

    def query_annotations_for(self, samples, gt_resource_id, ann_resource_id):
        self._maybe_update_graphs()
        noisy_samples = [
            sample.add_resource(
                self.__class__.__name__,
                ann_resource_id,
                self.apply_noise(sample.get_resource(gt_resource_id)),
            )
            for sample in samples
        ]

        return [
            sample
            for sample in noisy_samples
            if self.apply_filter(sample.get_resource(ann_resource_id))
        ]

    def apply_noise(self, uid):
        if self.noise_model == "Deng2014":
            noisy_uid = self._apply_deng_noise(uid)
        elif self.noise_model == "Geometric":
            noisy_uid = self._apply_geometric_noise(uid)
        elif self.noise_model == "Poisson":
            noisy_uid = self._apply_poisson_noise(uid)
        else:
            raise ValueError(f"Unknown noise model {self.noise_model}")
        print(f"Noise: {uid:20} -> {noisy_uid:20}")
        return noisy_uid

    def apply_filter(self, uid):
        if self.filter_imprecise:
            return self.graph.out_degree(uid) == 0  # noqa
        else:
            return True
