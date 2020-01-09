import random

from chia.framework.instrumentation import (
    InstrumentationContext,
    report,
    update_local_step,
)
from chia.framework import configuration, ioqueue
from chia.methods.incrementallearning.keras_incrementallearning import (
    KerasIncrementalModel,
)
from chia.data import sample
import time


class FastSingleShotKerasIncrementalModel(KerasIncrementalModel):
    def __init__(self, cls):
        KerasIncrementalModel.__init__(self, cls)

        with configuration.ConfigurationContext("FastSingleShotKerasIncrementalModel"):
            self._inner_steps = configuration.get("inner_steps", no_default=True)

        self._already_observed = False

    def observe_inner(self, samples, gt_resource_id, progress_callback=None):
        assert not self._already_observed, "This model can not learn continually"
        assert len(samples) > 0

        total_bs = (
            self.get_auto_batchsize(samples[0].get_resource("input_img_np").shape)
            * self.sequential_training_batches
        )

        def my_gen():
            for inner_step in range(self._inner_steps):
                batch_samples = random.choices(samples, k=total_bs)
                batch_elements_y = []
                batch_elements_X = []
                for sample in batch_samples:
                    batch_elements_X.append(_get_input_img_np(sample))
                    batch_elements_y.append(sample.get_resource(gt_resource_id))

                yield inner_step, (batch_elements_X, batch_elements_y)

        with InstrumentationContext(self.__class__.__name__):
            if progress_callback is not None:
                progress_callback(0.0)

            report("inner_steps", self._inner_steps)
            hc_loss_running = 0.0
            time_per_step_running = 0.0
            hc_loss_factor = 0.0
            last_step_end_time = time.time()
            for inner_step, (X, y) in ioqueue.make_generator_faster(
                my_gen, "threading", max_buffer_size=20
            ):
                if progress_callback is not None:
                    progress_callback(inner_step / float(self._inner_steps))
                update_local_step(inner_step)

                hc_loss = self.perform_single_gradient_step(X, y)

                hc_loss_running += hc_loss.numpy()
                hc_loss_factor += 1.0

                step_end_time = time.time()
                time_per_step_running += step_end_time - last_step_end_time

                if inner_step % 10 == 9:
                    report("loss_ravg", hc_loss_running / hc_loss_factor)
                    report("time_per_step", time_per_step_running / hc_loss_factor)
                    hc_loss_running = 0.0
                    hc_loss_factor = 0.0
                    time_per_step_running = 0.0

                last_step_end_time = step_end_time

            if progress_callback is not None:
                progress_callback(1.0)

    def rehearse(self, steps, progress_callback=None):
        raise ValueError("Cannot learn continually!")

    def save_inner(self, path):
        pass

    def restore_inner(self, path):
        pass


def _get_input_img_np(sample):
    return sample.get_resource("input_img_np")
