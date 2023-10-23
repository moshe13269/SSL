
from abc import ABC
import tensorflow as tf


class SmoothL1Loss(tf.keras.losses.Loss, ABC):
    beta: float

    def __init__(self, beta: float, tile_params):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.tile_params = tf.constant(tile_params)
        self.flatten = tf.keras.layers.Flatten()
        self.huber_loss = tf.keras.losses.Huber(delta=self.beta)
        # self.name = 'SmoothL1Loss'

    def call(self, y_true, y_pred):
        student_encoding, teacher_encoding = tf.split(y_pred, num_or_size_splits=2, axis=0)

        # teacher_encoding = self.flatten(teacher_encoding)
        # student_encoding = self.flatten(student_encoding)
        # y_true = tf.tile(y_true, self.tile_params)
        #
        # teacher_encoding = tf.boolean_mask(teacher_encoding, mask=y_true)
        # student_encoding = tf.boolean_mask(student_encoding, mask=y_true)
        #
        # return self.huber_loss(teacher_encoding, student_encoding)

        y_true = tf.expand_dims(y_true, axis=-1)
        huber = tf.keras.losses.Huber(delta=self.beta)
        return huber(teacher_encoding * y_true, student_encoding * y_true)
