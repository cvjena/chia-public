import abc
import pickle

import networkx


class Concept:
    def __init__(self, uid=None, data=None):
        if data is not None:
            self.data = data
            if uid is not None:
                raise ValueError("Cannot supply a UID if data is given.")
        else:
            if uid is not None:
                self.data = {"uid": uid}
            else:
                raise ValueError("Need uid for concept!")

    def __eq__(self, other):
        return self.data["uid"] == other.data["uid"]

    def __str__(self):
        return str(self.data)


class KnowledgeBase:
    def __init__(self):
        self.all_relations = {}
        self.all_concepts = {}
        self.concept_stamp = 0
        self.observation_stamp = 0
        self.relation_stamp = 0

    def is_known(self, concept_uid):
        return concept_uid in self.all_concepts.keys()

    def observe_concepts(self, concept_uids):
        new_concept = False
        for concept_uid in concept_uids:
            if concept_uid == "chia::UNCERTAIN":
                continue

            if not self.is_known(concept_uid):
                self.add_concept(concept_uid, dont_update_relations=True)
                new_concept = True

            self.all_concepts[concept_uid].data["observations"] += 1

        self.observation_stamp += 1

        if new_concept:
            self.update_all_relations(force_graph_update=True)

        return new_concept

    def observe_concept(self, concept_uid):
        return self.observe_concepts([concept_uid])

    def add_relation(
        self,
        uid,
        is_symmetric,
        is_transitive,
        is_reflexive,
        explore_left=False,
        explore_right=False,
        data=None,
        sources=None,
    ):
        relation = {
            "uid": uid,
            "data": data if data is not None else set(),
            "sources": sources if sources is not None else [],
            "graph": networkx.Graph() if is_symmetric else networkx.DiGraph(),
            "ugraph": networkx.Graph(),
            "is_reflexive": is_reflexive,
            "is_symmetric": is_symmetric,
            "is_transitive": is_transitive,
            "explore_left": explore_left,
            "explore_right": explore_right,
        }
        self.all_relations[uid] = relation

        self.update_relation(uid, True)

    def add_concept(self, uid=None, data=None, dont_update_relations=False):
        concept = Concept(uid=uid, data=data)
        concept.data["observations"] = 0
        self.all_concepts[concept.data["uid"]] = concept
        self.concept_stamp += 1

        if not dont_update_relations:
            self.update_all_relations(force_graph_update=True)

    def update_relation(self, uid, force_graph_update=False):
        made_changes_globally = force_graph_update
        relation = self.all_relations[uid]

        # Make it clear that everything else is not currently supported
        assert not relation["is_symmetric"]
        assert not relation["is_reflexive"]
        assert relation["is_transitive"]

        while True:
            made_changes_in_iteration = False
            updated_data = set()

            # Access sources
            if relation["explore_right"]:
                for concept_uid, concept in self.all_concepts.items():
                    for relation_source in relation["sources"]:
                        updated_data |= {
                            (concept_uid, right)
                            for right in relation_source.get_right_for(concept_uid)
                        } - relation["data"]

            if relation["explore_left"]:
                for concept_uid, concept in self.all_concepts.items():
                    for relation_source in relation["sources"]:
                        updated_data |= {
                            (left, concept_uid)
                            for left in relation_source.get_left_for(concept_uid)
                        } - relation["data"]

            if len(updated_data) > 0:
                relation["data"] |= updated_data
                made_changes_in_iteration = True

            # Update concepts
            updated_concepts = set()
            for (left, right) in relation["data"]:
                updated_concepts |= {left, right} - self.all_concepts.keys()

            if len(updated_concepts) > 0:
                for concept_uid in updated_concepts:
                    self.add_concept(
                        data={
                            "comment": "Automatically added by relation update",
                            "uid": concept_uid,
                        },
                        dont_update_relations=True,
                    )
                made_changes_in_iteration = True

            if not made_changes_in_iteration:
                break
            else:
                made_changes_globally = True

        # Update graph once
        if made_changes_globally:
            graph = relation["graph"]
            graph.update(nodes=self.all_concepts.keys(), edges=relation["data"])
            if relation["is_transitive"]:
                reduction = networkx.algorithms.dag.transitive_reduction(graph)
                relation["graph"] = reduction

            ugraph = relation["ugraph"]
            ugraph.update(nodes=self.all_concepts.keys(), edges=relation["data"])

            self.relation_stamp += 1

    def update_all_relations(self, force_graph_update=False):
        for relation_uid in self.all_relations.keys():
            self.update_relation(relation_uid, force_graph_update)

    def get_observed_concepts(self):
        return [
            concept
            for concept in self.all_concepts.values()
            if concept.data["observations"] > 0
        ]

    def get_concept_stamp(self):
        return self.concept_stamp

    def get_relation_stamp(self):
        return self.relation_stamp

    def get_observation_stamp(self):
        return self.observation_stamp

    def save(self, path):
        with open(path + "_knowledgebase.pkl", "wb") as target:
            pickle.dump(
                (
                    self.concept_stamp,
                    self.observation_stamp,
                    self.relation_stamp,
                    self.all_concepts,
                    self.all_relations,
                ),
                target,
            )

    def restore(self, path):
        with open(path + "_knowledgebase.pkl", "rb") as target:
            (
                self.concept_stamp,
                self.observation_stamp,
                self.relation_stamp,
                self.all_concepts,
                self.all_relations,
            ) = pickle.load(target)


class RelationSource(abc.ABC):
    @abc.abstractmethod
    def get_right_for(self, uid_left):
        pass

    @abc.abstractmethod
    def get_left_for(self, uid_right):
        pass


class StaticRelationSource(RelationSource):
    def __init__(self, data):
        self.right_for_left = {}
        self.left_for_right = {}
        for (left, right) in data:
            if left in self.right_for_left.keys():
                self.right_for_left[left] += [right]
            else:
                self.right_for_left[left] = [right]

            if right in self.left_for_right.keys():
                self.left_for_right[right] += [left]
            else:
                self.left_for_right[right] = [left]

    def get_left_for(self, uid_right):
        if uid_right in self.left_for_right.keys():
            return self.left_for_right[uid_right]
        else:
            return set()

    def get_right_for(self, uid_left):
        if uid_left in self.right_for_left.keys():
            return self.right_for_left[uid_left]
        else:
            return set()
