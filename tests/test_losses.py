import pytest
import tensorflow as tf
from pokemon_predictor.losses import FocalLoss

def test_focal_loss():
    y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)

    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    loss = loss_fn(y_true, y_pred)
    
    assert loss is not None
    assert loss.shape == () or loss.shape == (2,)
