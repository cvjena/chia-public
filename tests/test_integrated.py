import unittest

from chia.framework import configuration


class IntegratedTestCase(unittest.TestCase):
    def setUp(self):
        configuration.ConfigurationContext("integrated_test_case").__enter__()

    def test_integrated(self):
        label_gt_resource_id = "label_gt"
        label_ann_resource_id = "label_ann"
        label_pred_resource_id = "label_pred"

        from chia.data import datasets
        from chia.methods import (
            incrementallearning,
            hierarchicalclassification,
            interaction,
        )
        from chia import evaluation
        from chia import knowledge
        from chia.knowledge import wordnet

        dataset = datasets.dataset("iCIFAR")
        dataset.setup(classes_per_batch=100)
        assert dataset.train_pool_count() == 1

        with configuration.ConfigurationContext("KerasIncrementalModel"):
            configuration.set("use_pretrained_weights", None)
            configuration.set("train_feature_extractor", True)
            configuration.set("architecture", "keras::CIFAR-ResNet56")
            configuration.set("batchsize_max", 2)
            configuration.set("batchsize_min", 2)

        with configuration.ConfigurationContext("FastSingleShotKerasIncrementalModel"):
            configuration.set("inner_steps", 20)

        kb = knowledge.KnowledgeBase()
        im = interaction.method("Oracle", kb)
        ilm = incrementallearning.method(
            "keras::FastSingleShot",
            hierarchicalclassification.method("keras::OneHot", kb),
        )

        train_pool = dataset.train_pool(0, label_gt_resource_id)

        kb.observe_concepts(
            [sample.get_resource(label_gt_resource_id) for sample in train_pool]
        )

        train_pool = im.query_annotations_for(
            train_pool, label_gt_resource_id, label_ann_resource_id
        )

        test_pool = dataset.test_pool(0, label_gt_resource_id)
        test_pool = test_pool[: min(100, len(test_pool))]

        # Add hierarchy
        wna = wordnet.WordNetAccess()
        kb.add_relation(
            "hypernymy",
            is_symmetric=False,
            is_transitive=True,
            is_reflexive=False,
            explore_left=False,
            explore_right=True,
            sources=[dataset.relation("hypernymy"), wna],
        )

        # Train step
        ilm.observe(train_pool, label_ann_resource_id)

        # Prediction step
        test_predictions = ilm.predict(test_pool, label_pred_resource_id)

        # Evaluation
        evaluator = evaluation.method(evaluation.methods(), kb)
        evaluator.update(test_predictions, label_gt_resource_id, label_pred_resource_id)
        print(evaluator.result())

        pass


if __name__ == "__main__":
    unittest.main()
