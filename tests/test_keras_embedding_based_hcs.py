import unittest
import functools

from chia.methods import hierarchicalclassification
from chia import knowledge


class KerasEmbeddingBasedHCTestCase(unittest.TestCase):
    def setUp(self):
        self.kb = knowledge.KnowledgeBase()
        self.kb.observe_concepts(["A", "B", "C"])

        srs = knowledge.StaticRelationSource([("B", "A"), ("C", "A")])
        self.kb.add_relation(
            "hypernymy",
            is_reflexive=False,
            is_symmetric=False,
            is_transitive=True,
            sources=[srs],
        )

    def test_all_hcs(self):
        for method_key in hierarchicalclassification.methods():
            method = hierarchicalclassification.method(method_key, self.kb)
            with self.subTest("Testing HC", dataset=method_key):
                for concept in self.kb.get_observed_concepts():
                    embedded = method.embed([concept.data["uid"]])
                    deembedded = method.deembed(embedded)
                    self.assertEqual(concept.data["uid"], deembedded[0])

                    method.embed(["chia::EMPTY"])
