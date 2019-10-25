import tensorflow as tf
import tensorflow_addons as tfa
import math

from chia.framework import configuration


class KerasDataAugmentation:
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.do_random_flip = configuration.get("do_random_flip", True)

            self.do_random_rotate = configuration.get("do_random_rotate", True)

            self.do_random_crop = configuration.get("do_random_crop", True)
            self.random_crop_factor = configuration.get("random_crop_factor", 0.2)

            self.do_random_brightness_and_contrast = configuration.get(
                "do_random_brightness_and_contrast", True
            )
            self.random_brightness_factor = configuration.get(
                "random_brightness_factor", 0.05
            )
            self.random_contrast_factors = configuration.get(
                "random_contrast_factors", (0.7, 1.3)
            )

            self.do_random_hue_and_saturation = configuration.get(
                "do_random_hue_and_saturation", True
            )
            self.random_hue_factor = configuration.get("random_hue_factor", 0.08)
            self.random_saturation_factors = configuration.get(
                "random_saturation_factors", (0.6, 1.6)
            )

    @tf.function
    def process(self, sample_batch):
        sample_batch = tf.map_fn(self._process_sample, sample_batch)
        return sample_batch

    @tf.function
    def _process_sample(self, sample):
        if self.do_random_flip:
            sample = tf.image.random_flip_left_right(sample)
            sample = tf.image.random_flip_up_down(sample)

        if self.do_random_rotate:
            # sample = tf.image.rot90(
            #     sample, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            # )
            sample = tfa.image.rotate(
                sample,
                angles=tf.random.uniform(shape=[], minval=0, maxval=2.0 * math.pi),
                interpolation="BILINEAR",
            )

        if self.do_random_crop:
            crop_shape = sample.shape
            crop_px_h = (
                tf.cast(crop_shape[0] * self.random_crop_factor, dtype=tf.int32) // 2
            )
            sample = tf.pad(
                sample, [[crop_px_h, crop_px_h], [crop_px_h, crop_px_h], [0, 0]]
            )
            sample = tf.image.random_crop(sample, crop_shape)

        if self.do_random_brightness_and_contrast:
            sample = tf.image.random_brightness(sample, self.random_brightness_factor)
            sample = tf.image.random_contrast(sample, *self.random_contrast_factors)

        if self.do_random_hue_and_saturation:
            sample = tf.image.random_hue(sample, self.random_hue_factor)
            sample = tf.image.random_saturation(sample, *self.random_saturation_factors)

        return sample
