from abc import abstractmethod, ABC
import tensorflow as tf
import numpy as np
import pickle

from chia.framework.instrumentation import InstrumentationContext, report
from chia.framework import configuration


class KerasHierarchicalClassifier(ABC):
    @abstractmethod
    def predict(self, feature_batch):
        pass

    @abstractmethod
    def predict_dist(self, feature_batch):
        pass

    @abstractmethod
    def loss(self, feature_batch, ground_truth):
        pass

    @abstractmethod
    def observe(self, samples, gt_resource_id):
        pass

    @abstractmethod
    def regularization_losses(self):
        pass

    @abstractmethod
    def trainable_variables(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass


class EmbeddingBasedKerasHC(KerasHierarchicalClassifier, ABC):
    def __init__(self, kb):
        self.kb = kb
        self.last_observed_concept_stamp = kb.get_concept_stamp()

    @abstractmethod
    def predict_embedded(self, feature_batch):
        pass

    @abstractmethod
    def embed(self, labels):
        pass

    @abstractmethod
    def deembed_dist(self, embedded_labels):
        pass

    @abstractmethod
    def update_embedding(self):
        pass

    def predict(self, feature_batch):
        embedded_predictions = self.predict_embedded(feature_batch).numpy()
        return self.deembed(embedded_predictions)

    def predict_dist(self, feature_batch):
        embedded_predictions = self.predict_embedded(feature_batch).numpy()
        return self.deembed_dist(embedded_predictions)

    def deembed(self, embedded_labels):
        # TODO make nicer
        labels = []
        label_dists = self.deembed_dist(embedded_labels)
        for label_dist in label_dists:
            maxlabel = sorted(label_dist, key=lambda ld: ld[1], reverse=True)[0][0]
            labels.append(maxlabel)
        return labels

    def maybe_update_embedding(self):
        if self.kb.get_concept_stamp() != self.last_observed_concept_stamp:
            self.last_observed_concept_stamp = self.kb.get_concept_stamp()
            self.update_embedding()


class OneHotEmbeddingBasedKerasHC(EmbeddingBasedKerasHC):
    def __init__(self, kb):
        EmbeddingBasedKerasHC.__init__(self, kb)

        # Configuration
        with configuration.ConfigurationContext(self.__class__.__name__):
            self._l2_regularization_coefficient = configuration.get("l2", 5e-5)

        self.last_observed_concept_count = len(self.kb.get_observed_concepts())

        self.fc_layer = None
        self.uid_to_dimension = {}
        self.dimension_to_uid = []

        self.update_embedding()

    def update_embedding(self):
        with InstrumentationContext(self.__class__.__name__):
            current_observed_concepts = self.kb.get_observed_concepts()
            current_observed_concept_count = len(current_observed_concepts)

            # TODO take old weights
            try:
                old_weights = self.fc_layer.get_weights()
            except:
                old_weights = []

            self.fc_layer = tf.keras.layers.Dense(
                current_observed_concept_count,
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(
                    self._l2_regularization_coefficient
                ),
            )

            update_uids = [
                concept.data["uid"]
                for concept in current_observed_concepts
                if concept.data["uid"] not in self.dimension_to_uid
            ]

            self.dimension_to_uid += sorted(update_uids)

            self.uid_to_dimension = {
                uid: dimension for dimension, uid in enumerate(self.dimension_to_uid)
            }

            if len(old_weights) == 2:
                # Layer can be updated
                new_weights = np.concatenate(
                    [
                        old_weights[0],
                        np.zeros(
                            [
                                old_weights[0].shape[0],
                                current_observed_concept_count
                                - self.last_observed_concept_count,
                            ]
                        ),
                    ],
                    axis=1,
                )
                new_biases = np.concatenate(
                    [
                        old_weights[1],
                        np.zeros(
                            current_observed_concept_count
                            - self.last_observed_concept_count
                        ),
                    ],
                    axis=0,
                )

                self.fc_layer.build([None, old_weights[0].shape[0]])

                self.fc_layer.set_weights([new_weights, new_biases])
            report("observed_concepts", current_observed_concept_count)
            self.last_observed_concept_count = current_observed_concept_count

    def predict_embedded(self, feature_batch):
        return self.fc_layer(feature_batch)

    def embed(self, labels):
        dimensions = [self.uid_to_dimension[uid] for uid in labels]
        return tf.one_hot(dimensions, depth=self.last_observed_concept_count)

    def deembed_dist(self, embedded_labels):
        return [
            [(uid, embedded_label[dim]) for uid, dim in self.uid_to_dimension.items()]
            for embedded_label in embedded_labels
        ]

    def loss(self, feature_batch, ground_truth):
        embedded_predictions = self.predict_embedded(feature_batch)
        embedded_ground_truth = self.embed(ground_truth)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                embedded_ground_truth, embedded_predictions
            )
        )
        return loss

    def observe(self, samples, gt_resource_id):
        self.maybe_update_embedding()

    def regularization_losses(self):
        return self.fc_layer.losses

    def trainable_variables(self):
        return self.fc_layer.trainable_variables

    def save(self, path):
        with open(path + "_hc.pkl", "wb") as target:
            pickle.dump(self.fc_layer.get_weights(), target)

        with open(path + "_uidtodim.pkl", "wb") as target:
            pickle.dump((self.uid_to_dimension, self.dimension_to_uid), target)

    def restore(self, path):
        with open(path + "_hc.pkl", "rb") as target:
            new_weights = pickle.load(target)
            has_weights = False
            try:
                has_weights = len(self.fc_layer.get_weights()) == 2
            except:
                pass

            if not has_weights:
                self.fc_layer.build([None, new_weights[0].shape[0]])

            self.fc_layer.set_weights(new_weights)

        with open(path + "_uidtodim.pkl", "rb") as target:
            (self.uid_to_dimension, self.dimension_to_uid) = pickle.load(target)

        self.update_embedding()
