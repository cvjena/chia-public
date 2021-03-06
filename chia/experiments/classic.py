import math

from chia import evaluation, knowledge
from chia.data import datasets, pool
from chia.framework import configuration, instrumentation
from chia.knowledge import wordnet
from chia.methods import hierarchicalclassification, incrementallearning, interaction


@configuration.main_context()
def main():
    # Some constants
    label_gt_resource_id = "label_ground_truth"
    label_ann_resource_id = "label_annotated"
    label_pred_resource_id = "label_predicted"

    # General config
    ilm_method = configuration.get("ilm_method", no_default=True)
    cls_method = configuration.get("cls_method", no_default=True)
    interaction_method = configuration.get("interaction_method", no_default=True)
    experiment_scale = configuration.get("experiment_scale", no_default=True)
    dataset_name = configuration.get("dataset", no_default=True)
    experiment_name = configuration.get("experiment_name", no_default=True)
    report_interval = configuration.get("report_interval", no_default=True)
    report_initially = configuration.get("report_initially", no_default=True)
    validation_scale = configuration.get("validation_scale", no_default=True)
    skip_reclassification = configuration.get("skip_reclassification", no_default=True)
    evaluators = configuration.get("evaluators", no_default=True)

    # KB specific stuff
    observe_gt_concepts = configuration.get("observe_gt_concepts", no_default=True)

    # Save and restore
    restore_path = configuration.get("restore_path", no_default=True)
    save_path = configuration.get("save_path", no_default=True)
    save_path_append_run_number = configuration.get(
        "save_path_append_run_number", no_default=True
    )

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
        from chia.framework.instrumentation import sacred_instrumentation

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

            # Dataset specific run init
            if dataset_name == "CORe50":
                dataset.setup(scenario=core50_scenario, run=run_id)

            # Test datasets
            test_pools = []
            with instrumentation.InstrumentationContext("test_pools"):
                for i in range(dataset.test_pool_count()):
                    instrumentation.update_local_step(i)
                    test_pool = dataset.test_pool(i, label_gt_resource_id)
                    if validation_scale < 1.0:
                        test_pool = test_pool[
                            : min(
                                max(
                                    1, int(math.ceil(len(test_pool) * validation_scale))
                                ),
                                len(test_pool),
                            )
                        ]
                    instrumentation.report("size", len(test_pool))
                    test_pools += [test_pool]

            # Build methods
            kb = knowledge.KnowledgeBase()

            if restore_path is not None:
                kb.restore(restore_path)
            else:
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

            im = interaction.method(interaction_method, kb)
            cls = hierarchicalclassification.method(cls_method, kb)
            ilm = incrementallearning.method(ilm_method, cls)

            # Restore
            if restore_path is not None:
                ilm.restore(restore_path)

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

            if observe_gt_concepts:
                kb.observe_concepts(
                    [sample.get_resource(label_gt_resource_id) for sample in train_pool]
                )

            # Run "interaction"
            labeled_pool = im.query_annotations_for(
                train_pool, label_gt_resource_id, label_ann_resource_id
            )

            labeled_pool_size = len(labeled_pool)
            instrumentation.report("labeled_pool_size", labeled_pool_size)

            if report_initially:
                next_progress = 0.0
            else:
                next_progress = report_interval

            def evaluate(progress=None):
                nonlocal next_progress
                if progress is not None:
                    if progress < next_progress:
                        return
                    else:
                        next_progress += report_interval

                # Quick reclass accuracy
                if not skip_reclassification:
                    with instrumentation.InstrumentationContext(
                        "reclassification", take_time=True
                    ):
                        instrumentation.update_local_step(0)
                        if validation_scale < 1.0:
                            reclass_pool = labeled_pool[
                                : min(
                                    max(
                                        1,
                                        int(
                                            math.ceil(
                                                len(labeled_pool) * validation_scale
                                            )
                                        ),
                                    ),
                                    len(labeled_pool),
                                )
                            ]
                        else:
                            reclass_pool = labeled_pool
                        evaluator.update(
                            ilm.predict(reclass_pool, label_pred_resource_id),
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

                return results_across_test_pools

            with instrumentation.InstrumentationContext("training", take_time=True):

                # Learn the thing
                if not observe_gt_concepts:
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

            results_across_runs += [evaluate()]

            if save_path is not None and save_path_append_run_number:
                kb.save(f"{save_path}-{run_id}")
                ilm.save(f"{save_path}-{run_id}")

        instrumentation.store_result(results_across_runs)

        # Save last model
        if save_path is not None and not save_path_append_run_number:
            kb.save(save_path)
            ilm.save(save_path)


if __name__ == "__main__":
    main()
