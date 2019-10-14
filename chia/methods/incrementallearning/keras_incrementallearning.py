from abc import abstractmethod, ABC
import tensorflow as tf
import numpy as np
import random

from chia.methods.incrementallearning import ProbabilityOutputModel
from chia.instrumentation import report, update_local_step, InstrumentationContext
from chia.data.util import batches_from
from chia import configuration


class KerasIncrementalModel(ProbabilityOutputModel):
    def __init__(self, cls):
        self.cls = cls

        with configuration.ConfigurationContext(self.__class__.__name__):
            self.do_train_feature_extractor = configuration.get(
                "train_feature_extractor", False
            )
            self.exposure_coef = configuration.get("exposure_coef", 10)

        self.feature_extractor = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            input_tensor=None,
            input_shape=None,
            pooling="avg",
            weights="imagenet",
        )

        """
        self.feature_extractor = tf.keras.models.Sequential()
        self.feature_extractor.add(
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(None, None, 3),
            )
        )
        self.feature_extractor.add(tf.keras.layers.AveragePooling2D())
        self.feature_extractor.add(
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")
        )
        self.feature_extractor.add(tf.keras.layers.AveragePooling2D())
        self.feature_extractor.add(tf.keras.layers.Flatten())
        self.feature_extractor.add(tf.keras.layers.Dense(units=120, activation="relu"))
        self.feature_extractor.add(tf.keras.layers.Dense(units=84, activation="relu"))
        self.feature_extractor.add(
            tf.keras.layers.Dense(units=10, activation="softmax")
        )
        self.feature_extractor.summary()
        """

        # Add regularizer: see https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
        for layer in self.feature_extractor.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
                layer, tf.keras.layers.Dense
            ):
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(5e-5)(layer.kernel)
                )
            if hasattr(layer, "bias_regularizer") and layer.use_bias:
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(5e-5)(layer.bias)
                )

    @abstractmethod
    def observe_inner(self, samples, gt_resource_id):
        pass

    def observe(self, samples, gt_resource_id):
        self.cls.observe(samples, gt_resource_id)
        self.observe_inner(samples, gt_resource_id)

    def predict(self, samples, prediction_resource_id):
        return_samples = []
        for small_batch in batches_from(samples, batch_size=64):
            image_batch = self.preprocess_image_batch(
                self.build_image_batch(small_batch)
            )
            feature_batch = self.feature_extractor(image_batch, training=False)
            predictions = self.cls.predict(feature_batch)
            return_samples += [
                sample.add_resource(
                    self.__class__.__name__, prediction_resource_id, prediction
                )
                for prediction, sample in zip(predictions, small_batch)
            ]
        return return_samples

    def predict_probabilities(self, samples, prediction_dist_resource_id):
        return_samples = []
        for small_batch in batches_from(samples, batch_size=64):
            image_batch = self.preprocess_image_batch(
                self.build_image_batch(small_batch)
            )
            feature_batch = self.feature_extractor(image_batch, training=False)
            predictions = self.cls.predict_dist(feature_batch)
            return_samples += [
                sample.add_resource(
                    self.__class__.__name__, prediction_dist_resource_id, prediction
                )
                for prediction, sample in zip(predictions, small_batch)
            ]
        return return_samples

    def save(self, path):
        self.feature_extractor.save_weights(path)
        # TODO save and load classifier weights

    def restore(self, path):
        self.feature_extractor.load_weights(path)

    def build_image_batch(self, samples):
        assert len(samples) > 0
        return np.stack(
            [sample.get_resource("input_img_np") for sample in samples], axis=0
        )

    def preprocess_image_batch(self, image_batch):
        image_batch = tf.cast(image_batch, dtype=tf.float32)
        image_batch = (image_batch / 255.0) - 0.5
        return image_batch


class DFNKerasIncrementalModel(KerasIncrementalModel):
    def __init__(self, cls):
        KerasIncrementalModel.__init__(self, cls)
        self.X = []
        self.y = []

        self.optimizer = tf.keras.optimizers.Adam()

    def observe_inner(self, samples, gt_resource_id):
        assert len(samples) > 0

        total_bs = 128
        old_bs = 0
        new_bs = total_bs
        exposures = (
            len(samples)
            * self.exposure_coef
            / np.log10(len(samples))
            * (np.sqrt(1000) / np.sqrt(total_bs))
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

                batch_X = self.preprocess_image_batch(
                    np.stack(batch_elements_X, axis=0)
                )
                batch_y = (
                    batch_elements_y
                )  # No numpy stacking here, these could be strings or something else (concept uids)

                with tf.GradientTape() as tape:
                    feature_batch = self.feature_extractor(batch_X, training=True)
                    hc_loss = self.cls.loss(feature_batch, batch_y)
                    reg_loss = sum(
                        self.feature_extractor.losses + self.cls.regularization_losses()
                    )

                    total_loss = hc_loss + reg_loss

                if self.do_train_feature_extractor:
                    total_trainable_variables = (
                        self.feature_extractor.trainable_variables
                        + self.cls.trainable_variables()
                    )
                else:
                    total_trainable_variables = self.cls.trainable_variables()

                gradients = tape.gradient(total_loss, total_trainable_variables)

                self.optimizer.apply_gradients(
                    zip(gradients, total_trainable_variables)
                )

                if inner_step % 10 == 9:
                    report("loss", hc_loss.numpy())

            # Training done here
            for sample in samples:
                self.X.append(sample.get_resource("input_img_np"))
                self.y.append(sample.get_resource(gt_resource_id))

            report("storage", len(self.X))
