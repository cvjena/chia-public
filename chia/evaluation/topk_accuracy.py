import numpy as np

from chia import configuration

from . import Evaluator


class TopKAccuracyEvaluator(Evaluator):
    def __init__(self, kb):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.kmax = configuration.get("kmax", 5)

        self.kb = kb
        self.reset()

    def reset(self):
        self.sample_count = 0
        self.correct_count = np.zeros(self.kmax, dtype=np.float32)

    def update(self, samples, gt_resource_id, prediction_resource_id):
        for sample in iter(samples):
            if sample.has_resource(prediction_resource_id + "_dist"):
                self.sample_count += 1
                gt = sample.get_resource(gt_resource_id)
                pred = list(
                    sorted(
                        sample.get_resource(prediction_resource_id + "_dist"),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )[: self.kmax]
                pred = [x[0] for x in pred]

                for k in range(self.kmax):
                    if gt in pred[: k + 1]:
                        self.correct_count[k] += 1

    def result(self):
        if self.sample_count > 0:
            return {"top_k_acc": self.correct_count / float(self.sample_count)}
        else:
            return dict()
