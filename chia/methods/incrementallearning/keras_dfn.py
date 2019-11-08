import random
import math
import numpy as np
import tensorflow as tf

from chia.framework.instrumentation import (
    InstrumentationContext,
    report,
    update_local_step,
)
from chia.framework import configuration
from chia.methods.incrementallearning.keras_incrementallearning import (
    KerasIncrementalModel,
)


class DFNKerasIncrementalModel(KerasIncrementalModel):
    def __init__(self, cls):
        KerasIncrementalModel.__init__(self, cls)

        with configuration.ConfigurationContext("DFNKerasIncrementalModel"):
            self.exposure_coef = configuration.get("exposure_coef", 1.0)

        self.X = []
        self.y = []

    def observe_inner(self, samples, gt_resource_id):
        assert len(samples) > 0

        total_bs = self.get_auto_batchsize(
            samples[0].get_resource("input_img_np").shape
        )
        exposure_base = 3333 if not self.do_train_feature_extractor else 6666
        old_bs = 0
        new_bs = total_bs
        exposures = (
            len(samples)
            * self.exposure_coef
            / np.log10(len(samples))
            * (exposure_base / math.sqrt(total_bs))
        )
        inner_steps = int(max(1, exposures // total_bs))

        with InstrumentationContext(self.__class__.__name__):
            report("inner_steps", inner_steps)
            for inner_step in range(inner_steps):
                update_local_step(inner_step)

                batch_elements_X = []
                batch_elements_y = []

                # Old images
                if len(self.X) > 0:
                    old_indices = random.choices(range(len(self.X)), k=old_bs)
                    for old_index in old_indices:
                        batch_elements_X.append(self.X[old_index])
                        batch_elements_y.append(self.y[old_index])

                    old_bs = total_bs // 2
                    new_bs = total_bs - old_bs

                # New images
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

                if inner_step % 10 == 9:
                    report("loss", hc_loss.numpy())

            # Training done here
            for sample in samples:
                self.X.append(sample.get_resource("input_img_np"))
                self.y.append(sample.get_resource(gt_resource_id))

            report("storage", len(self.X))
