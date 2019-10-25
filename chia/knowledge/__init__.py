import pickle


class Concept:
    def __init__(self, uid=None, data=None):
        if data is not None:
            self.data = data
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
        self.all_concepts = {}
        self.concept_stamp = 0
        # TODO observation stamp?

    def is_known(self, concept_uid):
        return concept_uid in self.all_concepts.keys()

    def observe_concept(self, concept_uid):
        new_concept = False
        if not self.is_known(concept_uid):
            self.add_concept(concept_uid)
            new_concept = True

        self.all_concepts[concept_uid].data["observations"] += 1
        return new_concept

    def add_concept(self, uid, data=None):
        concept = Concept(uid=uid, data=data)
        concept.data["observations"] = 0
        self.all_concepts[uid] = concept
        self.concept_stamp += 1

    def get_observed_concepts(self):
        return [
            concept
            for concept in self.all_concepts.values()
            if concept.data["observations"] > 0
        ]

    def get_concept_stamp(self):
        return self.concept_stamp

    def save(self, path):
        with open(path + "_knowledgebase.pkl", "wb") as target:
            pickle.dump((self.concept_stamp, self.all_concepts), target)

    def restore(self, path):
        with open(path + "_knowledgebase.pkl", "rb") as target:
            self.concept_stamp, self.all_concepts = pickle.load(target)
