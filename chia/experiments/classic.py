from chia.data import datasets
from chia.methods import (
    incrementallearning,
    hierarchicalclassification,
    activelearning,
    interaction,
)
from chia.framework.instrumentation import sacred_instrumentation
from chia import knowledge
from chia.knowledge import wordnet
from chia import evaluation
from chia.framework import configuration, instrumentation
from chia.data import pool

import math


@configuration.main_context()
def main():
    # Some constants
    label_gt_resource_id = "label_ground_truth"
    label_ann_resource_id = "label_annotated"
    label_pred_resource_id = "label_predicted"

    # General config
    ilm_method = configuration.get("ilm_method", no_default=True)
    cls_method = configuration.get("cls_method", no_default=True)
    experiment_scale = configuration.get("experiment_scale", no_default=True)
    dataset_name = configuration.get("dataset", no_default=True)
    experiment_name = configuration.get("experiment_name", no_default=True)
    report_interval = configuration.get("report_interval", no_default=True)
    evaluators = configuration.get("evaluators", no_default=True)

    # Eval config
    use_sacred_observer = configuration.get("use_sacred_observer", no_default=True)

    # Dataset specific setup stuff
    if dataset_name == "CORe50":
        core50_scenario = configuration.get("core50_scenario", no_default=True)

    # Instantiate dataset
    dataset = datasets.dataset(dataset_name)

    # Get instrumentation going
    instrumentation_observers = [
        instrumentation.PrintObserver(experiment_name),
        instrumentation.JSONResultObserver(experiment_name),
    ]
    if use_sacred_observer:
        instrumentation_observers += [
            sacred_instrumentation.SacredObserver(experiment_name)
        ]

    with instrumentation.InstrumentationContext(
        "run", observers=instrumentation_observers
    ):
        # Determine run count
        if dataset_name == "CORe50":
            run_count = dataset.get_run_count(core50_scenario)
        else:
            run_count = configuration.get("run_count", no_default=True)

        run_count = max(1, int(math.ceil(run_count * experiment_scale)))
        instrumentation.report("run_count", run_count)

        results_across_runs = []

        # Runs...
        for run_id in range(run_count):
            results_during_run = []
            instrumentation.update_local_step(run_id)

            # Dataset specific run init
            if dataset_name == "CORe50":
                dataset.setup(scenario=core50_scenario, run=run_id)

            # Test datasets
            test_pools = []
            with instrumentation.InstrumentationContext("test_pools"):
                for i in range(dataset.test_pool_count()):
                    instrumentation.update_local_step(i)
                    test_pool = dataset.test_pool(i, label_gt_resource_id)

                    instrumentation.report("size", len(test_pool))
                    test_pools += [test_pool]

            # Build methods
            kb = knowledge.KnowledgeBase()
            im = interaction.OracleInteractionMethod()

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
            cls = hierarchicalclassification.method(cls_method, kb)
            ilm = incrementallearning.method(ilm_method, cls)

            # Evaluator
            evaluator = evaluation.method(evaluators, kb)

            train_pool_count = dataset.train_pool_count()

            # Collect training data
            train_pool = []
            for train_pool_id in range(train_pool_count):
                train_pool += pool.FixedPool(
                    dataset.train_pool(train_pool_id, label_gt_resource_id)
                )

            train_pool_size = len(train_pool)
            instrumentation.report("train_pool_size", train_pool_size)

            # Run "interaction"
            labeled_pool = im.query_annotations_for(
                train_pool, label_gt_resource_id, label_ann_resource_id
            )

            next_progress = 0.0

            def evaluate(progress=None):
                nonlocal next_progress
                nonlocal results_during_run
                if progress is not None:
                    if progress < next_progress:
                        return
                    else:
                        next_progress += report_interval

                # Quick reclass accuracy
                with instrumentation.InstrumentationContext("reclassification"):
                    evaluator.update(
                        ilm.predict(labeled_pool, label_pred_resource_id),
                        label_ann_resource_id,
                        label_pred_resource_id,
                    )
                    instrumentation.report_dict(evaluator.result())
                    evaluator.reset()

                # Validation
                with instrumentation.InstrumentationContext(
                    "validation", take_time=True
                ):
                    results_across_test_pools = []
                    for test_pool_id in range(len(test_pools)):
                        instrumentation.update_local_step(test_pool_id)
                        evaluator.update(
                            ilm.predict(
                                test_pools[test_pool_id], label_pred_resource_id
                            ),
                            label_gt_resource_id,
                            label_pred_resource_id,
                        )
                        instrumentation.report_dict(evaluator.result())
                        results_across_test_pools += [evaluator.result()]
                        evaluator.reset()

                if progress is not None:
                    results_during_run += [results_across_test_pools]
                return results_across_test_pools

            with instrumentation.InstrumentationContext("training", take_time=True):

                # Learn the thing
                kb.observe_concepts(
                    [
                        sample.get_resource(label_ann_resource_id)
                        for sample in labeled_pool
                    ]
                )
                ilm.observe(
                    labeled_pool,
                    label_ann_resource_id,
                    progress_callback=evaluate if report_interval > 0 else None,
                )

            results_across_runs += [results_during_run + evaluate()]

        instrumentation.store_result(results_across_runs)


if __name__ == "__main__":
    main()
