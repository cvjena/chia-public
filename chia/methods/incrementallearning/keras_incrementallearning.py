import functools
import math
import multiprocessing
import pickle as pkl
import time
from abc import abstractmethod

import numpy as np

import tensorflow as tf
from chia.data.util import batches_from, batches_from_pair
from chia.framework import configuration, instrumentation, ioqueue
from chia.methods.common import keras_dataaugmentation, keras_learningrateschedule
from chia.methods.hierarchicalclassification import keras_hierarchicalclassification
from chia.methods.incrementallearning import ProbabilityOutputModel
from tensorflow.keras.applications import (
    inception_resnet_v2,
    mobilenet_v2,
    nasnet,
    resnet_v2,
)


class KerasIncrementalModel(ProbabilityOutputModel):
    def __init__(
        self, cls: keras_hierarchicalclassification.KerasHierarchicalClassifier
    ):
        self.cls = cls

        with configuration.ConfigurationContext("KerasIncrementalModel"):
            # Preprocessing
            self.random_crop_to_size = configuration.get("random_crop_to_size", None)
            _channel_mean = configuration.get("channel_mean", [127.5, 127.5, 127.5])
            self.channel_mean_normalized = np.array(_channel_mean) / 255.0
            _channel_stddev = configuration.get("channel_stddev", [127.5, 127.5, 127.5])
            self.channel_stddev_normalized = np.array(_channel_stddev) / 255.0

            # Batch size
            self.batchsize_max = configuration.get("batchsize_max", 256)
            self.batchsize_min = configuration.get("batchsize_min", 1)
            self.sequential_training_batches = configuration.get(
                "sequential_training_batches", 1
            )
            self.autobs_vram = configuration.get(
                "autobs_vram", configuration.get_system("gpu0_vram")
            )

            # Fine-tuning options
            self.do_train_feature_extractor = configuration.get(
                "train_feature_extractor", False
            )
            self.use_pretrained_weights = configuration.get(
                "use_pretrained_weights", "ILSVRC2012"
            )

            # Architecture
            self.architecture = configuration.get("architecture", "keras::ResNet50V2")

            # Optimization and regularization
            self.l2_regularization = configuration.get("l2_regularization", 5e-5)
            self.optimizer_name = configuration.get("optimizer", "adam")
            if self.optimizer_name == "sgd":
                self.sgd_momentum = configuration.get("sgd_momentum", 0.9)
            self.lr_schedule_cfg = configuration.get(
                "lr_schedule", {"name": "constant", "config": {"initial_lr": 0.003}}
            )
            self.lr_schedule = keras_learningrateschedule.get(self.lr_schedule_cfg)

        if self.architecture == "keras::ResNet50V2":
            self.feature_extractor = resnet_v2.ResNet50V2(
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
            self.feature_extractor = inception_resnet_v2.InceptionResNetV2(
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
            self.feature_extractor = mobilenet_v2.MobileNetV2(
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
            self.feature_extractor = nasnet.NASNetMobile(
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

        if (
            self.use_pretrained_weights is not None
            and self.use_pretrained_weights != "ILSVRC2012"
        ):
            print(
                f"Loading alternative pretrained weights {self.use_pretrained_weights}"
            )
            self.feature_extractor.load_weights(self.use_pretrained_weights)

        if not self.do_train_feature_extractor:
            for layer in self.feature_extractor.layers:
                layer.trainable = False

        self.reported_auto_bs = False

        # State here
        self.current_step = 0

    def _add_regularizers(self):
        if self.l2_regularization == 0:
            return

        # Add regularizer:
        # see https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
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

        total_time_data = 0.0
        total_time_preprocess = 0.0
        total_time_features = 0.0
        total_time_cls = 0.0
        total_time_write = 0.0

        def my_gen():
            pool = multiprocessing.pool.ThreadPool(4)
            for small_batch_ in batches_from(samples, batch_size=auto_bs):
                built_image_batch_ = self.build_image_batch(small_batch_, pool)
                yield small_batch_, built_image_batch_

        tp_before_data = time.time()
        faster_generator = ioqueue.make_generator_faster(
            my_gen, method="threading", max_buffer_size=50
        )
        for (small_batch, built_image_batch) in faster_generator:
            tp_before_preprocess = time.time()
            image_batch = self.preprocess_image_batch(
                built_image_batch, is_training=False
            )
            tp_before_features = time.time()
            feature_batch = self.feature_extractor(image_batch, training=False)
            tp_before_cls = time.time()
            predictions_dist = self.cls.predict_dist(feature_batch)
            predictions = [
                sorted(prediction_dist, key=lambda x: x[1], reverse=True)[0][0]
                for prediction_dist in predictions_dist
            ]

            tp_before_write = time.time()
            return_samples += [
                sample.add_resource(
                    self.__class__.__name__, prediction_resource_id, prediction
                ).add_resource(
                    self.__class__.__name__,
                    prediction_resource_id + "_dist",
                    prediction_dist,
                )
                for prediction, prediction_dist, sample in zip(
                    predictions, predictions_dist, small_batch
                )
            ]
            tp_loop_done = time.time()
            total_time_data += tp_before_preprocess - tp_before_data
            total_time_preprocess += tp_before_features - tp_before_preprocess
            total_time_features += tp_before_cls - tp_before_features
            total_time_cls += tp_before_write - tp_before_cls
            total_time_write += tp_loop_done - tp_before_write

            tp_before_data = time.time()

        print("Predict done.")
        print(f"Time (data): {total_time_data}")
        print(f"Time (preprocess): {total_time_preprocess}")
        print(f"Time (features): {total_time_features}")
        print(f"Time (cls): {total_time_cls}")
        print(f"Time (write): {total_time_write}")
        total_time_overall = (
            total_time_data
            + total_time_preprocess
            + total_time_features
            + total_time_cls
            + total_time_write
        )
        print(f"Total time: {total_time_overall}")
        return return_samples

    def predict_probabilities(self, samples, prediction_dist_resource_id):
        return_samples = []
        auto_bs = self.get_auto_batchsize(samples[0].get_resource("input_img_np").shape)

        def my_gen():
            pool = multiprocessing.pool.ThreadPool(4)
            for small_batch_ in batches_from(samples, batch_size=auto_bs):
                built_image_batch_ = self.build_image_batch(small_batch_, pool)
                yield small_batch_, built_image_batch_

        for small_batch, built_image_batch in ioqueue.make_generator_faster(
            my_gen, method="synchronous", max_buffer_size=5
        ):
            image_batch = self.preprocess_image_batch(
                built_image_batch, is_training=False
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

    def build_image_batch(self, samples, pool=None):
        assert len(samples) > 0
        if pool is not None:
            np_images = pool.map(_get_input_img_np, samples)
        else:
            np_images = [sample.get_resource("input_img_np") for sample in samples]
        return np.stack(np_images, axis=0)

    def preprocess_image_batch(self, image_batch, is_training, processing_fn=None):
        image_batch = tf.cast(image_batch, dtype=tf.float32) / 255.0
        if processing_fn is not None:
            # Processing_fn expects values in [0, 1]
            image_batch = processing_fn(image_batch)

        # Map to correct range, e.g. [-1.0 , 1.0]
        image_batch = image_batch - self.channel_mean_normalized
        image_batch = image_batch / self.channel_stddev_normalized

        # Do cropping here instead of in augmentation because all augmentation is
        # disabled during testing...
        if self.random_crop_to_size is not None:
            if is_training:
                image_batch = tf.map_fn(self._random_crop_single_image, image_batch)
            else:
                image_batch = tf.image.crop_to_bounding_box(
                    image_batch,
                    (image_batch.shape[1] - self.random_crop_to_size[1]) // 2,
                    (image_batch.shape[2] - self.random_crop_to_size[0]) // 2,
                    self.random_crop_to_size[1],
                    self.random_crop_to_size[0],
                )

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
        if self.do_train_feature_extractor:
            total_trainable_variables = (
                self.feature_extractor.trainable_variables
                + self.cls.trainable_variables()
            )
        else:
            total_trainable_variables = self.cls.trainable_variables()

        # Forward step
        inner_bs = self.get_auto_batchsize(batch_elements_X[0].shape)
        acc_gradients = None
        acc_hc_loss = 0
        inner_batch_count = 0

        for inner_batch_X, inner_batch_y in batches_from_pair(
            batch_elements_X, batch_elements_y, inner_bs
        ):
            # Build batch
            batch_X = self.preprocess_image_batch(
                np.stack(inner_batch_X, axis=0),
                is_training=True,
                processing_fn=self.augmentation.process
                if self.augmentation is not None
                else None,
            )
            batch_y = inner_batch_y
            # No numpy stacking here, these could be
            # strings or something else (concept uids)

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

            # Backward step
            gradients = tape.gradient(total_loss, total_trainable_variables)
            if acc_gradients is None:
                acc_gradients = gradients
            else:
                acc_gradients = [
                    acc_gradient + new_gradient
                    for acc_gradient, new_gradient in zip(acc_gradients, gradients)
                ]

            acc_hc_loss += hc_loss
            inner_batch_count += 1

        # Optimize
        self.optimizer.learning_rate = self.lr_schedule(self.current_step)
        self.optimizer.apply_gradients(
            zip(
                [
                    acc_gradient / float(inner_batch_count)
                    for acc_gradient in acc_gradients
                ],
                total_trainable_variables,
            )
        )

        self.current_step += 1
        return acc_hc_loss / float(inner_batch_count)

    def _random_crop_single_image(self, image):
        return tf.image.random_crop(
            image, [self.random_crop_to_size[1], self.random_crop_to_size[0], 3]
        )


def _get_input_img_np(sample):
    return sample.get_resource("input_img_np")
