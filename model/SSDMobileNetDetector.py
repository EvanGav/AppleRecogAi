from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model.BaseDetector import BaseDetector
from model.DetectorModel import DetectorModel
from model.assignLabel import assignLabelsToGrid

class SSDMobileNetDetector(BaseDetector):
    """
    Anchor-free grid detector on MobileNetV2.

    Uses a single-scale 20×20 grid from block_13_expand_relu.
    """

    GRID_SIZES:  List[Tuple[int, int]] = [] 
    TOTAL_CELLS: int = 0

    def __init__(
        self,
        conf_threshold:    float = 0.3,
        nms_iou_threshold: float = 0.25,
        max_detections:    int   = 100,
    ) -> None:
        self._model:       Optional[DetectorModel] = None
        self._conf_thresh: float = conf_threshold
        self._nms_iou:     float = nms_iou_threshold
        self._max_dets:    int   = max_detections

    # ── BaseDetector ───────────────────────────────────────────────────────────

    def Build(self, num_classes: int, input_shape: Tuple[int, int, int]) -> None:
        """Build the model and print a summary."""
        functional = self._BuildFunctionalModel(input_shape)
        self._model = DetectorModel(functional)
        self._model.compile(optimizer=keras.optimizers.Adam(1e-3))
        # Initialise weights with one dummy forward pass
        dummy = np.zeros((1,) + input_shape, dtype=np.float32)
        self._model(dummy)
        self._model.summary(line_length=100)
        print(f"\n  Grid sizes  : {self.GRID_SIZES}")
        print(f"  Total cells : {self.TOTAL_CELLS}")

    def Train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset:   tf.data.Dataset,
        epochs: int = 20,
    ) -> keras.callbacks.History:
        """Two-phase training."""
        if self._model is None:
            raise RuntimeError("Call Build() first.")

        train_ds  = self._ReformatDataset(train_dataset)
        val_ds    = self._ReformatDataset(val_dataset)
        callbacks = self._BuildCallbacks()

        # Strategy: freeze the entire backbone and only train the small detection head
        # because MinneApple has only ~530 training images. The MobileNetV2 backbone
        # has 2.2M parameters. Even with regularisation, unfreezing any backbone
        # layers causes the model to memorise training images within 1-2 epochs
        # /!\ This strategy was given to me by Claude after searching for a long time and trying many things

        print("\n=== Phase 1: head only, backbone FULLY frozen (LR=1e-3) ===")
        # Freeze the entire backbone (all layers) for the whole training run — unfreezing even a few layers causes overfitting on MinneApple's small dataset
        self._FreezeBackbone(True)
        self._model.compile(optimizer=keras.optimizers.Adam(1e-3))
        warmup = min(15, max(5, epochs * 2 // 3))
        history = self._model.fit(
            train_ds, validation_data=val_ds,
            epochs=warmup, callbacks=callbacks,
        )
        if epochs > warmup:
            print("\n=== Phase 2: head refinement, backbone still frozen (LR=2e-4) ===")
            self._model.compile(optimizer=keras.optimizers.Adam(2e-4))
            h2 = self._model.fit(
                train_ds, validation_data=val_ds,
                epochs=epochs, initial_epoch=warmup,
                callbacks=callbacks,
            )
            for k, v in h2.history.items():
                history.history.setdefault(k, []).extend(v)

        return history

    def Predict(self, image: np.ndarray, conf_override: float = None) -> Dict[str, np.ndarray]:
        """Decode model output into boxes/scores.

        Parameters :
        conf_override : optional threshold override (e.g. use 0.1 for mAP eval
                        to maximise recall across the full precision-recall curve)
        """
        if self._model is None:
            raise RuntimeError("Call Build() and Load() first.")

        conf = conf_override if conf_override is not None else self._conf_thresh

        batch  = np.expand_dims(image.astype(np.float32), 0)
        output = self._model.predict(batch, verbose=0)[0]   # (cells, 5)

        obj_logits = output[:, 0]    # (cells,)
        box_preds  = output[:, 1:]   # (cells, 4) — cy, cx, h, w  absolute normalised

        scores = tf.sigmoid(obj_logits).numpy()

        def _Sig(x): return 1.0 / (1.0 + np.exp(-x))
        cy = _Sig(box_preds[:, 0])
        cx = _Sig(box_preds[:, 1])
        h  = _Sig(box_preds[:, 2]) * 0.5
        w  = _Sig(box_preds[:, 3]) * 0.5
        boxes = np.stack([cy - h/2, cx - w/2, cy + h/2, cx + w/2], axis=1)
        boxes = np.clip(boxes, 0.0, 1.0)

        keep = scores >= conf
        if not np.any(keep):
            return {
                "boxes":  np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros(0,      dtype=np.float32),
                "labels": np.zeros(0,      dtype=np.int32),
            }

        boxes  = boxes[keep]
        scores = scores[keep]

        selected = tf.image.non_max_suppression(
            boxes, scores,
            max_output_size=self._max_dets,
            iou_threshold=self._nms_iou,
            score_threshold=conf,
        ).numpy()

        return {
            "boxes":  boxes[selected],
            "scores": scores[selected],
            "labels": np.ones(len(selected), dtype=np.int32),
        }

    def Save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_weights(str(path))
        print(f"Saved → {path}")

    def Load(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("Call Build() first.")
        self._model.load_weights(str(path))
        print(f"Loaded ← {path}")

    # ── Private: model construction ────────────────────────────────────────────

    def _BuildFunctionalModel(
        self,
        input_shape: Tuple[int, int, int],
    ) -> keras.Model:
        """
        Build the functional Keras model.
        """
        inputs = keras.Input(shape=input_shape, name="image")

        backbone = keras.applications.MobileNetV2(
            input_tensor=inputs, include_top=False, weights="imagenet",
        )

        # Tap two feature scales
        feat_coarse = backbone.get_layer("block_13_expand_relu").output
        feat_fine   = backbone.get_layer("block_6_expand_relu").output

        # Read actual shapes (never hardcode — they depend on input_shape)
        ch, cw = int(feat_coarse.shape[1]), int(feat_coarse.shape[2])
        fh, fw = int(feat_fine.shape[1]),   int(feat_fine.shape[2])

        self.__class__.GRID_SIZES  = [(ch, cw), (fh, fw)]
        self.__class__.TOTAL_CELLS = ch * cw + fh * fw

        def Head(features, h: int, w: int, tag: str) -> tf.Tensor:
            """
            Lightweight detection head for one feature scale.
            Returns (batch, h*w, 5).
            """
            # Shared conv backbone
            x = keras.layers.Conv2D(
                32, 3, padding="same", activation="relu",
                name=f"{tag}_c1",
            )(features)
            x = keras.layers.BatchNormalization(name=f"{tag}_bn")(x)
            x = keras.layers.Dropout(0.5, name=f"{tag}_drop")(x)

            # Objectness branch — careful initialisation to avoid gradient collapse
            obj = keras.layers.Conv2D(
                1, 1, name=f"{tag}_obj",
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=keras.initializers.Zeros(),
                kernel_regularizer=keras.regularizers.L2(5e-4),
            )(x)   # (batch, h, w, 1)

            # Box branch
            box = keras.layers.Conv2D(
                4, 1, name=f"{tag}_box",
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=keras.initializers.Zeros(),
                kernel_regularizer=keras.regularizers.L2(5e-4),
            )(x)   # (batch, h, w, 4)

            # Stack channels then reshape to (batch, h*w, 5)
            combined = keras.layers.Concatenate(axis=-1, name=f"{tag}_cat")([obj, box])
            return keras.layers.Reshape((h * w, 5), name=f"{tag}_out")(combined)

        out_coarse = Head(feat_coarse, ch, cw, "coarse")
        out_fine   = Head(feat_fine,   fh, fw, "fine")

        # All cells from both scales: (batch, total_cells, 5)
        combined = keras.layers.Concatenate(axis=1, name="all_cells")(
            [out_coarse, out_fine]
        )

        return keras.Model(inputs=inputs, outputs=combined, name="AppleDetector")

    def _ReformatDataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Convert dataset from padded-box format to per-cell target format.
        """
        total = self.TOTAL_CELLS
        grid  = self.GRID_SIZES   # captured for closure

        # Runs once per batch and applies the expensive assignLabelsToGrid function in a tf.py_function to convert GT boxes/labels into cell-wise targets 
        def _MapFn(images, labels):
            gt_boxes  = labels["box_output"]    # (B, 50, 4)
            gt_labels = labels["class_output"]  # (B, 50)

            targets, = tf.py_function(
                func=lambda b, l: [self._AssignBatch(b, l)],
                inp=[gt_boxes, gt_labels],
                Tout=[tf.float32],
            )
            targets.set_shape([None, total, 5])
            return images, targets

        return ds.map(_MapFn, num_parallel_calls=1)

    def _AssignBatch(
        self,
        gt_boxes_t:  tf.Tensor,
        gt_labels_t: tf.Tensor,
    ) -> np.ndarray:
        """Run AssignLabelsToGrid for every image in the batch."""
        boxes_np  = gt_boxes_t.numpy()
        labels_np = gt_labels_t.numpy()
        B = boxes_np.shape[0]

        out = np.zeros((B, self.TOTAL_CELLS, 5), dtype=np.float32)
        for i in range(B):
            out[i] = assignLabelsToGrid(boxes_np[i], labels_np[i], self.GRID_SIZES)
        return out

    def _FreezeBackbone(self, frozen: bool, freeze_until: int = 0) -> None:
        """Freeze/unfreeze the MobileNetV2 backbone inside the model."""
        if self._model is None:
            return
        try:
            backbone = self._model._net.get_layer("mobilenetv2_1.00_320")
        except ValueError:
            return
        if frozen:
            backbone.trainable = False
        else:
            backbone.trainable = True
            for layer in backbone.layers[:freeze_until]:
                layer.trainable = False

    def _BuildCallbacks(self) -> List[keras.callbacks.Callback]:
        """Standard callbacks."""
        Path("checkpoints").mkdir(exist_ok=True)
        return [
            keras.callbacks.ModelCheckpoint(
                filepath="checkpoints/best_weights.weights.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(log_dir="logs/", update_freq="epoch"),
        ]
