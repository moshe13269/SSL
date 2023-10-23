import tensorflow as tf
from tslearn.metrics import soft_dtw


class SoftDTWLoss(tf.keras.losses.Loss):

    def __init__(self, gamma):
        super(SoftDTWLoss, self).__init__()
        self.gamma = gamma

    def call(self, y_true, y_pred):
        soft_dtw_score = soft_dtw(y_true, y_pred, gamma=.1)
        return soft_dtw_score
