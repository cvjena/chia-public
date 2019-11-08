import pickle

import numpy as np
import tensorflow as tf
import networkx as nx

from chia.framework import configuration
from chia.framework.instrumentation import InstrumentationContext, report
from chia.methods.hierarchicalclassification.keras_hierarchicalclassification import (
    EmbeddingBasedKerasHC,
)


class IDKEmbeddingBasedKerasHC(EmbeddingBasedKerasHC):
    def __init__(self, kb):
        EmbeddingBasedKerasHC.__init__(self, kb)

        # Configuration
        with configuration.ConfigurationContext("IDKEmbeddingBasedKerasHC"):
            self._l2_regularization_coefficient = configuration.get("l2", 5e-5)
            self._mlnp = configuration.get("mlnp", True)

        self.fc_layer = None
        self.uid_to_dimension = {}
        self.graph = None
        self.observed_uids = None
        self.topo_sorted_uids = None
        self.update_embedding()

    def predict_embedded(self, feature_batch):
        return self.fc_layer(feature_batch)

    def embed(self, labels):
        embedding = np.zeros((len(labels), len(self.uid_to_dimension)))
        for i, label in enumerate(labels):
            embedding[i, self.uid_to_dimension[label]] = 1.0
            for ancestor in nx.ancestors(self.graph, label):
                embedding[i, self.uid_to_dimension[ancestor]] = 1.0

        return embedding

    def deembed_dist(self, embedded_labels):
        return [
            self._deembed_single(embedded_label) for embedded_label in embedded_labels
        ]

    def _deembed_single(self, embedded_label):
        conditional_probabilities = {
            uid: embedded_label[i] for uid, i in self.uid_to_dimension.items()
        }
        # Stage 1 calculates the unconditional probabilities
        unconditional_probabilities = {}

        # Stage 2 calculates the joint probability of the synset and "no children"
        joint_probabilities = {}

        for uid in self.topo_sorted_uids:
            unconditional_probability = conditional_probabilities[uid]

            no_parent_probability = 1.0
            has_parents = False
            for parent in self.graph.predecessors(uid):
                has_parents = True
                no_parent_probability *= 1.0 - unconditional_probabilities[parent]

            if has_parents:
                unconditional_probability *= 1.0 - no_parent_probability

            unconditional_probabilities[uid] = unconditional_probability

        for uid in reversed(self.topo_sorted_uids):
            joint_probability = unconditional_probabilities[uid]
            no_child_probability = 1.0
            for child in self.graph.successors(uid):
                no_child_probability *= 1.0 - unconditional_probabilities[child]

            joint_probabilities[uid] = joint_probability * no_child_probability

        tuples = joint_probabilities.items()
        sorted_tuples = list(sorted(tuples, key=lambda tup: tup[1], reverse=True))

        if self._mlnp:
            for i, (uid, p) in enumerate(sorted_tuples):
                if uid not in self.observed_uids:
                    sorted_tuples[i] = (uid, 0.0)

        return list(sorted_tuples)

    def update_embedding(self):
        with InstrumentationContext(self.__class__.__name__):
            current_concepts = self.kb.all_concepts.values()
            current_concept_count = len(current_concepts)
            report("current_concepts", current_concept_count)

        self.fc_layer = tf.keras.layers.Dense(
            current_concept_count,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(
                self._l2_regularization_coefficient
            ),
        )

        # We need to reverse the graph for comfort because "is-a" has the concepts
        self.graph = self.kb.all_relations["hypernymy"]["graph"].reverse(copy=True)

        # Memorize topological sorting for later
        all_uids = nx.topological_sort(self.graph)
        self.topo_sorted_uids = list(all_uids)
        assert len(self.kb.all_concepts) == len(self.topo_sorted_uids)

        self.uid_to_dimension = {
            uid: dimension for dimension, uid in enumerate(self.topo_sorted_uids)
        }

        self.observed_uids = {
            concept.data["uid"] for concept in self.kb.get_observed_concepts()
        }

    def loss(self, feature_batch, ground_truth):
        loss_mask = np.zeros((len(ground_truth), len(self.uid_to_dimension)))
        for i, label in enumerate(ground_truth):
            # Loss mask
            loss_mask[i, self.uid_to_dimension[label]] = 1.0

            for ancestor in nx.ancestors(self.graph, label):
                loss_mask[i, self.uid_to_dimension[ancestor]] = 1.0
                for successor in self.graph.successors(ancestor):
                    loss_mask[i, self.uid_to_dimension[successor]] = 1.0
                    # This should also cover the node itself, but we do it anyway

            for successor in self.graph.successors(label):
                loss_mask[i, self.uid_to_dimension[successor]] = 1.0

        embedding = self.embed(ground_truth)
        prediction = self.predict_embedded(feature_batch)

        # Binary cross entropy loss function
        clipped_probs = tf.clip_by_value(prediction, 1e-7, (1.0 - 1e-7))
        the_loss = -(
            embedding * tf.math.log(clipped_probs)
            + (1.0 - embedding) * tf.math.log(1.0 - clipped_probs)
        )

        sum_per_batch_element = tf.reduce_sum(the_loss * loss_mask, axis=1)
        return tf.reduce_mean(sum_per_batch_element)

    def observe(self, samples, gt_resource_id):
        if self.kb.get_concept_stamp() != self.last_observed_concept_stamp:
            self.update_embedding()
            self.last_observed_concept_stamp = self.kb.get_concept_stamp()

    def regularization_losses(self):
        return self.fc_layer.losses

    def trainable_variables(self):
        return self.fc_layer.trainable_variables

    def save(self, path):
        with open(path + "_hc.pkl", "wb") as target:
            pickle.dump(self.fc_layer.get_weights(), target)

        with open(path + "_uidtodim.pkl", "wb") as target:
            pickle.dump((self.uid_to_dimension,), target)

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
            (self.uid_to_dimension,) = pickle.load(target)

        # TODO uncomment this when incremental learning starts working
        # self.update_embedding()