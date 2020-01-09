import math

import tensorflow as tf
import tensorflow_addons as tfa

from chia.framework import configuration


class KerasDataAugmentation:
    def __init__(self):
        with configuration.ConfigurationContext(self.__class__.__name__):
            self.do_random_flip_horizontal = configuration.get(
                "do_random_flip_horizontal", True
            )
            self.do_random_flip_vertical = configuration.get(
                "do_random_flip_vertical", True
            )

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

            self.do_random_scale = configuration.get("do_random_scale", True)
            self.random_scale_factors = configuration.get(
                "random_scale_factors", (0.5, 2.0)
            )

    @tf.function
    def process(self, sample_batch):
        sample_batch = tf.map_fn(self._process_sample, sample_batch)
        return sample_batch

    @tf.function
    def _process_sample(self, sample):
        if self.do_random_flip_horizontal:
            sample = tf.image.random_flip_left_right(sample)
        if self.do_random_flip_vertical:
            sample = tf.image.random_flip_up_down(sample)

        if self.do_random_rotate:
            sample = tfa.image.rotate(
                sample,
                angles=tf.random.uniform(shape=[], minval=0, maxval=2.0 * math.pi),
                interpolation="BILINEAR",
            )

        if self.do_random_crop or self.do_random_scale:
            sample = self._inner_random_crop_or_scale(sample)

        if self.do_random_brightness_and_contrast:
            sample = tf.image.random_brightness(sample, self.random_brightness_factor)
            sample = tf.image.random_contrast(sample, *self.random_contrast_factors)

        if self.do_random_hue_and_saturation:
            sample = tf.image.random_hue(sample, self.random_hue_factor)
            sample = tf.image.random_saturation(sample, *self.random_saturation_factors)

        return sample

    @tf.function
    def _inner_random_crop_or_scale(self, x: tf.Tensor) -> tf.Tensor:
        if self.do_random_scale:
            scale = tf.random.uniform(
                shape=[],
                minval=self.random_scale_factors[0],
                maxval=self.random_scale_factors[1],
            )
        else:
            scale = 1.0

        width = 1.0 / scale
        height = 1.0 / scale

        if self.do_random_crop:
            left_min = -self.random_crop_factor
            left_max = 1.0 + self.random_crop_factor - width
            top_min = -self.random_crop_factor
            top_max = 1.0 + self.random_crop_factor - height
        else:
            left_min = 0.0
            left_max = 1.0 - width
            top_min = 0.0
            top_max = 1.0 - height

        left = tf.random.uniform(shape=[], minval=left_min, maxval=left_max)
        top = tf.random.uniform(shape=[], minval=top_min, maxval=top_max)
        boxes = [[top, left, top + height, left + width]]

        crop_shape = (x.shape[0], x.shape[1])
        # Create different crops for an image
        crops = tf.image.crop_and_resize(
            [x], boxes=boxes, box_indices=[0], crop_size=crop_shape
        )
        # Return a random crop
        return crops[0]
