from abc import abstractmethod, ABC


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
