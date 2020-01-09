import math
import os
import pickle as pkl
import random

import numpy as np

from chia.data import sample
from chia.framework import configuration
from chia.framework.instrumentation import (
    InstrumentationContext,
    report,
    update_local_step,
)
from chia.methods.incrementallearning.keras_incrementallearning import (
    KerasIncrementalModel,
)


class DFNKerasIncrementalModel(KerasIncrementalModel):
    def __init__(self, cls):
        KerasIncrementalModel.__init__(self, cls)

        with configuration.ConfigurationContext("DFNKerasIncrementalModel"):
            self.exposure_coef = configuration.get("exposure_coef", 1.0)
            self.always_rehearse = configuration.get("always_rehearse", False)
            self.lambda_ = configuration.get("lambda", 0.5)
            self.fixed_inner_steps = configuration.get("fixed_inner_steps", None)

        self.rehearsal_pool = []

    def observe_inner(self, samples, gt_resource_id, progress_callback=None):
        assert len(samples) > 0

        total_bs = (
            self.get_auto_batchsize(samples[0].get_resource("input_img_np").shape)
            * self.sequential_training_batches
        )
        exposure_base = 2500 if not self.do_train_feature_extractor else 10000
        new_bs = total_bs
        exposures = (
            len(samples)
            * self.exposure_coef
            / np.log10(len(samples))
            * (exposure_base / math.sqrt(total_bs))
        )
        if self.fixed_inner_steps is None:
            inner_steps = int(max(1, exposures // total_bs))
        else:
            inner_steps = self.fixed_inner_steps

        with InstrumentationContext(self.__class__.__name__):
            if progress_callback is not None:
                progress_callback(0.0)

            rehearse_only = False
            report("inner_steps", inner_steps)
            hc_loss_running = 0.0
            hc_loss_factor = 0.0
            for inner_step in range(inner_steps):
                if progress_callback is not None:
                    progress_callback(inner_step / float(inner_steps))
                update_local_step(inner_step)

                batch_elements_X = []
                batch_elements_y = []

                # Old images
                if len(self.rehearsal_pool) > 0:
                    if rehearse_only:
                        old_bs = total_bs
                        new_bs = 0
                    else:
                        old_bs = max(
                            0, min(total_bs, int(math.ceil(total_bs * self.lambda_)))
                        )
                        new_bs = total_bs - old_bs
                    old_indices = random.choices(
                        range(len(self.rehearsal_pool)), k=old_bs
                    )
                    for old_index in old_indices:
                        batch_elements_X.append(
                            self.rehearsal_pool[old_index].get_resource("input_img_np")
                        )
                        batch_elements_y.append(
                            self.rehearsal_pool[old_index].get_resource("zDFN.label")
                        )

                # New images
                if not rehearse_only:
                    new_indices = random.choices(range(len(samples)), k=new_bs)
                    for new_index in new_indices:
                        batch_elements_X.append(
                            samples[new_index].get_resource("input_img_np")
                        )
                        batch_elements_y.append(
                            samples[new_index].get_resource(gt_resource_id)
                        )

                hc_loss = self.perform_single_gradient_step(
                    batch_elements_X, batch_elements_y
                )

                hc_loss_running += hc_loss.numpy()
                hc_loss_factor += 1.0

                if inner_step % 10 == 9:
                    report("loss_ravg", hc_loss_running / hc_loss_factor)
                    hc_loss_running = 0.0
                    hc_loss_factor = 0.0

                if (
                    not rehearse_only
                    and inner_step >= inner_steps // 2
                    and self.always_rehearse
                ):
                    print("Switching to rehearsal mode.")
                    self.add_to_rehearsal_pool(gt_resource_id, samples)

                    rehearse_only = True

            if not self.always_rehearse:
                print("Rehearsal disabled.")
                self.add_to_rehearsal_pool(gt_resource_id, samples)

            report("storage", len(self.rehearsal_pool))

            if progress_callback is not None:
                progress_callback(1.0)

    def add_to_rehearsal_pool(self, gt_resource_id, samples):
        for sample_ in samples:
            self.rehearsal_pool.append(
                sample_.add_resource(
                    self.__class__.__name__,
                    "zDFN.label",
                    sample_.get_resource(gt_resource_id),
                )
            )

    def rehearse(self, steps, progress_callback=None):
        assert len(self.rehearsal_pool) > 0

        if progress_callback is not None:
            progress_callback(0.0)

        total_bs = self.get_auto_batchsize(
            self.rehearsal_pool[0].get_resource("input_img_np").shape
        )

        with InstrumentationContext(self.__class__.__name__):
            report("inner_steps", steps)
            hc_loss_running = 0.0
            hc_loss_factor = 0.0
            for inner_step in range(steps):
                if progress_callback is not None:
                    progress_callback(inner_step / float(steps))
                update_local_step(inner_step)

                batch_elements_X = []
                batch_elements_y = []

                old_indices = random.choices(
                    range(len(self.rehearsal_pool)), k=total_bs
                )
                for old_index in old_indices:
                    batch_elements_X.append(
                        self.rehearsal_pool[old_index].get_resource("input_img_np")
                    )
                    batch_elements_y.append(
                        self.rehearsal_pool[old_index].get_resource("zDFN.label")
                    )

                hc_loss = self.perform_single_gradient_step(
                    batch_elements_X, batch_elements_y
                )

                hc_loss_running += hc_loss.numpy()
                hc_loss_factor += 1.0

                if inner_step % 10 == 9:
                    report("loss_ravg", hc_loss_running / hc_loss_factor)
                    hc_loss_running = 0.0
                    hc_loss_factor = 0.0

            report("storage", len(self.rehearsal_pool))

        if progress_callback is not None:
            progress_callback(1.0)

    def save_inner(self, path):
        with open(path + "_dfnpool.pkl", "wb") as target:
            pkl.dump(self.rehearsal_pool, target)

    def restore_inner(self, path):
        if not os.path.exists(path + "_dfnpool.pkl"):
            print("Falling back to dfnstate.pkl")
            with open(path + "_dfnstate.pkl", "rb") as target:
                (X, y) = pkl.load(target)
                for i, (Xi, yi) in enumerate(zip(X, y)):
                    self.rehearsal_pool.append(
                        sample.Sample(self.__class__.__name__, uid=f"DFNImport_{i:05d}")
                        .add_resource(self.__class__.__name__, "input_img_np", Xi)
                        .add_resource(self.__class__.__name__, "zDFN.label", yi)
                    )
        else:
            with open(path + "_dfnpool.pkl", "rb") as target:
                self.rehearsal_pool = pkl.load(target)
