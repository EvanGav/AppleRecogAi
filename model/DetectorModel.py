
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras

class DetectorModel(keras.Model):
    """
    Keras Model subclass.

    Holds the backbone + head and implements a custom train_step that
    computes the masked detection loss correctly.

    Single output: (batch, total_cells, 5)
    """

    # Box-loss weight relative to objectness BCE
    BOX_LAMBDA: float = 0.5   # gentle weight while sigmoid h/w converges

    def __init__(self, functional_model: keras.Model) -> None:
        super().__init__()
        self._net = functional_model   # the functional model built below

    def call(self, inputs, training=False):
        return self._net(inputs, training=training)

    def _DetectionLoss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute total, objectness, and box losses.

        Parameters
        ----------
        y_true : (batch, cells, 5)  col0=obj, cols1-4=box targets
        y_pred : (batch, cells, 5)  col0=obj logit, cols1-4=box preds

        Returns
        -------
        total_loss, obj_loss, box_loss
        """
        obj_true  = y_true[..., 0]      # (batch, cells)
        box_true  = y_true[..., 1:]     # (batch, cells, 4)
        obj_logit = y_pred[..., 0]      # (batch, cells)
        box_pred  = y_pred[..., 1:]     # (batch, cells, 4)

        # Objectness loss — BCE on ALL cells 
        # Use from_logits=True for numerical stability
        obj_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(obj_true, tf.float32),
                logits=obj_logit,
            )
        )

        # Box loss — MSE on positive cells only
        pos_mask = tf.cast(obj_true > 0.5, tf.float32)      # (batch, cells)
        pos_mask_4d = tf.expand_dims(pos_mask, -1)          # (batch, cells, 1)

        # sqrt/sigmoid^2 encoding for h/w — no exp() so no explosion:
        #   cy, cx: sigmoid(pred) vs target          (both in [0,1])
        #   h,  w:  sigmoid(pred) vs sqrt(target)    (sqrt maps 0.05→0.22)
        # At inference: h = sigmoid(raw)^2 recovers the real h.
        pred_cy = tf.sigmoid(box_pred[..., 0:1])
        pred_cx = tf.sigmoid(box_pred[..., 1:2])
        pred_h  = tf.sigmoid(box_pred[..., 2:3])
        pred_w  = tf.sigmoid(box_pred[..., 3:4])
        true_cy = box_true[..., 0:1]
        true_cx = box_true[..., 1:2]
        # Scale targets to [0,1] by dividing by 0.5 (max expected apple size)
        # sigmoid output * 0.5 at decode -> h/w in [0, 0.5]
        true_h  = tf.clip_by_value(box_true[..., 2:3], 0.0, 1.0) / 0.5
        true_w  = tf.clip_by_value(box_true[..., 3:4], 0.0, 1.0) / 0.5
        box_pred_enc = tf.concat([pred_cy, pred_cx, pred_h, pred_w], axis=-1)
        box_true_enc = tf.concat([true_cy, true_cx, true_h, true_w], axis=-1)
        sq_err = tf.square(box_true_enc - box_pred_enc)  # (batch, cells, 4)
        masked  = sq_err * pos_mask_4d # zero negatives
        n_pos   = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
        box_loss = tf.reduce_sum(masked) / n_pos

        total = obj_loss + self.BOX_LAMBDA * box_loss
        return total, obj_loss, box_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            total, obj_loss, box_loss = self._DetectionLoss(y, y_pred)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total, "obj_loss": obj_loss, "box_loss": box_loss}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        total, obj_loss, box_loss = self._DetectionLoss(y, y_pred)
        return {"loss": total, "obj_loss": obj_loss, "box_loss": box_loss}
