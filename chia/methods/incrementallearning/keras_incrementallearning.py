from abc import abstractmethod
import tensorflow as tf
import numpy as np
import functools
import math
import pickle as pkl

from chia.methods.common import keras_dataaugmentation, keras_learningrateschedule
from chia.methods.incrementallearning import ProbabilityOutputModel
from chia.data.util import batches_from
from chia.framework import configuration, instrumentation


class KerasIncrementalModel(ProbabilityOutputModel):
    def __init__(self, cls):
        self.cls = cls

        with configuration.ConfigurationContext("KerasIncrementalModel"):
            self.do_train_feature_extractor = configuration.get(
                "train_feature_extractor", False
            )
            self.use_pretrained_weights = configuration.get(
                "use_pretrained_weights", "ILSVRC2012"
            )
            self.batchsize_max = configuration.get("batchsize_max", 256)
            self.batchsize_min = configuration.get("batchsize_min", 1)
            self.autobs_vram = configuration.get(
                "autobs_vram", configuration.get_system("gpu0_vram")
            )

            self.architecture = configuration.get("architecture", "keras::ResNet50V2")
            self.l2_regularization = configuration.get("l2_regularization", 5e-5)
            self.optimizer_name = configuration.get("optimizer", "adam")
            if self.optimizer_name == "sgd":
                self.sgd_momentum = configuration.get("sgd_momentum", 0.9)
            self.lr_schedule_cfg = configuration.get(
                "lr_schedule", {"name": "constant", "config": {"initial_lr": 0.003}}
            )
            self.lr_schedule = keras_learningrateschedule.get(self.lr_schedule_cfg)

        if self.architecture == "keras::ResNet50V2":
            self.feature_extractor = tf.keras.applications.resnet_v2.ResNet50V2(
                include_top=False,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )
            self.pixels_per_gb = 1100000

            self._add_regularizers()

        elif self.architecture == "keras::InceptionResNetV2":
            self.feature_extractor = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
                include_top=False,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )
            self.pixels_per_gb = 700000

            self._add_regularizers()

        elif self.architecture == "keras::MobileNetV2":
            with configuration.ConfigurationContext("KerasIncrementalModel"):
                self.side_length = configuration.get("side_length", no_default=True)
            self.feature_extractor = tf.keras.applications.mobilenet_v2.MobileNetV2(
                include_top=False,
                input_tensor=None,
                input_shape=(self.side_length, self.side_length, 3),
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )
            self.pixels_per_gb = 2000000

            self._add_regularizers()

        elif self.architecture == "keras::NASNetMobile":
            with configuration.ConfigurationContext("KerasIncrementalModel"):
                self.side_length = configuration.get("side_length", no_default=True)
            self.feature_extractor = tf.keras.applications.nasnet.NASNetMobile(
                include_top=False,
                input_tensor=None,
                input_shape=(self.side_length, self.side_length, 3),
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )
            self.pixels_per_gb = 1350000

            self._add_regularizers()

        elif self.architecture == "keras::CIFAR-ResNet56":
            assert (
                self.do_train_feature_extractor
            ), "There are no pretrained weights for this architecture!"
            assert (
                self.use_pretrained_weights is None
            ), "There are no pretrained weights for this architecture!"

            from chia.methods.common import keras_cifar_resnet

            self.feature_extractor = keras_cifar_resnet.feature_extractor(
                version=2, n=6, l2_norm=self.l2_regularization
            )
            self.pixels_per_gb = 200000

        else:
            raise ValueError(f'Unknown architecture "{self.architecture}"')

        if self.optimizer_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(self.lr_schedule(0))
        else:
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.lr_schedule(0), momentum=self.sgd_momentum
            )
        self.augmentation = keras_dataaugmentation.KerasDataAugmentation()

        if not self.do_train_feature_extractor:
            for layer in self.feature_extractor.layers:
                layer.trainable = False

        self.reported_auto_bs = False

        # State here
        self.current_step = 0

    def _add_regularizers(self):
        if self.l2_regularization == 0:
            return

        # Add regularizer: see https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
        for layer in self.feature_extractor.layers:
            if (
                isinstance(layer, tf.keras.layers.Conv2D)
                and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)
            ) or isinstance(layer, tf.keras.layers.Dense):
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(
                        self.l2_regularization
                    )(layer.kernel)
                )
            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(
                        self.l2_regularization
                    )(layer.depthwise_kernel)
                )
            if hasattr(layer, "bias_regularizer") and layer.use_bias:
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(
                        self.l2_regularization
                    )(layer.bias)
                )

    @abstractmethod
    def observe_inner(self, samples, gt_resource_id, progress_callback=None):
        pass

    def save_inner(self, path):
        pass

    def restore_inner(self, path):
        pass

    def observe(self, samples, gt_resource_id, progress_callback=None):
        self.cls.observe(samples, gt_resource_id)
        self.observe_inner(samples, gt_resource_id, progress_callback)

    def predict(self, samples, prediction_resource_id):
        return_samples = []
        auto_bs = self.get_auto_batchsize(samples[0].get_resource("input_img_np").shape)

        for small_batch in batches_from(samples, batch_size=auto_bs):
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
        auto_bs = self.get_auto_batchsize(samples[0].get_resource("input_img_np").shape)

        for small_batch in batches_from(samples, batch_size=auto_bs):
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
        self.feature_extractor.save_weights(path + "_features.h5")
        with open(path + "_ilstate.pkl", "wb") as target:
            pkl.dump(self.current_step, target)
        self.save_inner(path)
        self.cls.save(path)

    def restore(self, path):
        self.feature_extractor.load_weights(path + "_features.h5")
        with open(path + "_ilstate.pkl", "rb") as target:
            self.current_step = pkl.load(target)
        self.restore_inner(path)
        self.cls.restore(path)

    def build_image_batch(self, samples):
        assert len(samples) > 0
        return np.stack(
            [sample.get_resource("input_img_np") for sample in samples], axis=0
        )

    def preprocess_image_batch(self, image_batch, processing_fn=None):
        if processing_fn is not None:
            image_batch = tf.cast(image_batch, dtype=tf.float32) / 255.0
            image_batch = processing_fn(image_batch)
            image_batch = (2.0 * image_batch) - 1.0
            return image_batch
        else:
            image_batch = tf.cast(image_batch, dtype=tf.float32)
            image_batch = (image_batch / 127.5) - 1.0
            return image_batch

    def get_auto_batchsize(self, shape):
        # Calculate input buffer size
        input_buffer_bytes = 4.0 * functools.reduce(lambda x, y: x * y, shape[:2], 1)

        # Magic AUTO BS formula
        auto_bs = math.floor(
            (self.pixels_per_gb / input_buffer_bytes) * self.autobs_vram
        )

        # Clip to reasonable values
        auto_bs = min(self.batchsize_max, max(self.batchsize_min, auto_bs))

        # Report
        if not self.reported_auto_bs:
            with instrumentation.InstrumentationContext(self.__class__.__name__):
                instrumentation.report("auto_bs", auto_bs)
            self.reported_auto_bs = True

        return auto_bs

    def perform_single_gradient_step(self, batch_elements_X, batch_elements_y):
        # Build batch
        batch_X = self.preprocess_image_batch(
            np.stack(batch_elements_X, axis=0),
            processing_fn=self.augmentation.process
            if self.augmentation is not None
            else None,
        )
        batch_y = (
            batch_elements_y
        )  # No numpy stacking here, these could be strings or something else (concept uids)

        # Forward step
        with tf.GradientTape() as tape:
            feature_batch = self.feature_extractor(
                batch_X, training=self.do_train_feature_extractor
            )
            hc_loss = self.cls.loss(feature_batch, batch_y)

            if self.do_train_feature_extractor:
                reg_loss = sum(
                    self.feature_extractor.losses + self.cls.regularization_losses()
                )
            else:
                reg_loss = self.cls.regularization_losses()

            total_loss = hc_loss + reg_loss
        if self.do_train_feature_extractor:
            total_trainable_variables = (
                self.feature_extractor.trainable_variables
                + self.cls.trainable_variables()
            )
        else:
            total_trainable_variables = self.cls.trainable_variables()

        # Backward step
        gradients = tape.gradient(total_loss, total_trainable_variables)

        # Optimize
        self.optimizer.learning_rate = self.lr_schedule(self.current_step)
        self.optimizer.apply_gradients(zip(gradients, total_trainable_variables))

        self.current_step += 1
        return hc_loss
