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
    al_score_resource_id = "al_score"

    # General config
    ilm_method = configuration.get("ilm_method", no_default=True)
    cls_method = configuration.get("cls_method", no_default=True)
    al_method = configuration.get("al_method", no_default=True)
    experiment_scale = configuration.get("experiment_scale", no_default=True)
    label_budget = configuration.get("label_budget", no_default=True)
    dataset_name = configuration.get("dataset", no_default=True)
    experiment_name = configuration.get("experiment_name", no_default=True)
    evaluators = configuration.get("evaluators", no_default=True)

    ll_cycle_mode = configuration.get("ll_cycle_mode", no_default=True)
    if ll_cycle_mode:
        ll_cycle_length = configuration.get("ll_cycle_length", no_default=True)
    else:
        ll_cycle_length = 0

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
            instrumentation.update_local_step(run_id)
            results_across_train_pools = []

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
            alm = activelearning.method(al_method, ilm, kb)

            # Evaluator
            evaluator = evaluation.method(evaluators, kb)

            # Go over batches...
            with instrumentation.InstrumentationContext("train_pool"):
                train_pool_count = dataset.train_pool_count()
                instrumentation.report("train_pool_count", train_pool_count)

                for train_pool_id in range(train_pool_count):
                    instrumentation.update_local_step(train_pool_id)

                    with instrumentation.InstrumentationContext(
                        "training", take_time=True
                    ):
                        # Collect training data
                        train_pool = pool.FixedPool(
                            dataset.train_pool(train_pool_id, label_gt_resource_id)
                        )

                        train_pool_size = len(train_pool)
                        instrumentation.report("train_pool_size", train_pool_size)
                        train_pool_label_budget = max(
                            1, math.ceil(label_budget * train_pool_size)
                        )
                        instrumentation.report(
                            "train_pool_budget", train_pool_label_budget
                        )

                        # Start the cycle through the training data
                        with instrumentation.InstrumentationContext("llcycle"):
                            current_cycle = 0
                            while train_pool_label_budget > 0:
                                instrumentation.update_local_step(current_cycle)
                                assert train_pool_label_budget <= len(train_pool)

                                if ll_cycle_mode:
                                    current_cycle_budget = min(
                                        train_pool_label_budget, ll_cycle_length
                                    )
                                else:
                                    current_cycle_budget = train_pool_label_budget

                                instrumentation.report(
                                    "current_cycle_budget", current_cycle_budget
                                )
                                instrumentation.report(
                                    "train_pool_size", len(train_pool)
                                )
                                train_pool_label_budget -= current_cycle_budget

                                # Only do active learning if there is even a choice
                                if (len(train_pool) - current_cycle_budget) > 0:
                                    # Run active learning method
                                    scored_train_pool = alm.score(
                                        train_pool, al_score_resource_id
                                    )
                                    sorted_scored_train_pool = list(
                                        sorted(
                                            scored_train_pool,
                                            key=lambda sample: sample.get_resource(
                                                al_score_resource_id
                                            ),
                                            reverse=True,
                                        )
                                    )

                                    pool_to_be_labeled = sorted_scored_train_pool[
                                        :current_cycle_budget
                                    ]

                                    instrumentation.report(
                                        "min_al_score",
                                        sorted_scored_train_pool[-1].get_resource(
                                            al_score_resource_id
                                        ),
                                    )
                                    instrumentation.report(
                                        "cutoff_al_score",
                                        pool_to_be_labeled[-1].get_resource(
                                            al_score_resource_id
                                        ),
                                    )
                                    instrumentation.report(
                                        "max_al_score",
                                        pool_to_be_labeled[0].get_resource(
                                            al_score_resource_id
                                        ),
                                    )
                                else:
                                    pool_to_be_labeled = train_pool

                                # Run "interaction"
                                labeled_pool = im.query_annotations_for(
                                    pool_to_be_labeled,
                                    label_gt_resource_id,
                                    label_ann_resource_id,
                                )

                                # Learn the thing
                                kb.observe_concepts(
                                    [
                                        sample.get_resource(label_ann_resource_id)
                                        for sample in labeled_pool
                                    ]
                                )
                                ilm.observe(labeled_pool, label_ann_resource_id)

                                # Remove the labeled samples from the pool
                                samples_before_removal = len(train_pool)
                                train_pool = train_pool.remove_multiple(labeled_pool)
                                assert (
                                    len(train_pool)
                                    == samples_before_removal - current_cycle_budget
                                )
                                current_cycle += 1

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

                    results_across_train_pools += [results_across_test_pools]

            results_across_runs += results_across_train_pools

        instrumentation.store_result(results_across_runs)


if __name__ == "__main__":
    main()
