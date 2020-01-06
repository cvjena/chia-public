import numpy as np
from networkx.algorithms import shortest_paths
import networkx as nx


class SemanticMeasure:
    def __init__(self, kb):
        self.kb = kb
        self.last_concept_stamp = -1
        self.graph = None
        self.ugraph = None
        self.measure_cache = None
        self._update_measure_cache()

    def _maybe_update_measure_cache(self):
        if self.last_concept_stamp != self.kb.get_concept_stamp():
            self._update_measure_cache()

    def _update_measure_cache(self):
        self.last_concept_stamp = self.kb.get_concept_stamp()
        relation = self.kb.all_relations["hypernymy"]
        concept_uids = self.kb.all_concepts.keys()
        self.graph = relation["graph"].reverse(copy=True)
        self.ugraph = relation["ugraph"]

        self.measure_cache = {}
        for uida in concept_uids:
            self.measure_cache[uida] = {}
            for uidb in concept_uids:
                self.measure_cache[uida][uidb] = self._measure_inner(uida, uidb)

    def _measure_inner(self, uida, uidb):
        return 1

    def measure(self, uida, uidb):
        self._maybe_update_measure_cache()
        return self.measure_cache[uida][uidb]


class Rada1989SemanticMeasure(SemanticMeasure):
    def _measure_inner(self, uida, uidb):
        return shortest_paths.shortest_path_length(self.ugraph, uida, uidb)


class Zhong2002SemanticMeasure(SemanticMeasure):
    def _measure_inner(self, uida, uidb):
        return NotImplementedError("Need to implement tree reduction first")


class MaedcheStaab2001SemanticMeasure(SemanticMeasure):
    def _measure_inner(self, uida, uidb):
        feature_set_x = nx.ancestors(self.graph, source=uida) | {uida}
        feature_set_y = nx.ancestors(self.graph, source=uidb) | {uidb}
        sim = float(len(feature_set_x & feature_set_y)) / float(
            len(feature_set_x | feature_set_y)
        )
        return (1.0 / sim) - 1.0


_method_mapping = {
    "Rada1989": Rada1989SemanticMeasure,
    "Zhong2002": Zhong2002SemanticMeasure,
    "MaedcheStaab2001": MaedcheStaab2001SemanticMeasure,
}


def methods():
    return _method_mapping.keys()


def method(key, kb):
    return _method_mapping[key](kb)
