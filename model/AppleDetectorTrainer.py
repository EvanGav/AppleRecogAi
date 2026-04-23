from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from model.BaseDetector   import BaseDetector
from data.BaseDataLoader import BaseDataLoader
from utils.MaskAnnotationParser import BoundingBox
from utils.metrics    import ComputeMeanAveragePrecision


class AppleDetectorTrainer:
    """
    Coordinator that drives training and evaluation.

    Parameters :
    detector    : any BaseDetector implementation
    data_loader : any BaseDataLoader implementation
    output_dir  : folder where checkpoints and final weights are saved
    """

    def __init__(
        self,
        detector:    BaseDetector,
        data_loader: BaseDataLoader,
        output_dir:  Path = Path("output"),
    ) -> None:
        self._detector    = detector
        self._data_loader = data_loader
        self._output_dir  = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def PrepareAndTrain(self, epochs: int = 20) -> Dict[str, Any]:
        """
        Full pipeline: build model -> load data -> train -> evaluate -> save

        Returns :
        dict with keys 'history' and 'metrics'
        """
        # Derive input shape from target size used by the data loader
        # We inspect the first batch to get the actual image shape
        train_ds = self._data_loader.GetTrainDataset()
        val_ds   = self._data_loader.GetValidationDataset()

        sample_batch = next(iter(train_ds))
        input_shape  = tuple(sample_batch[0].shape[1:])  # (H, W, C)

        print(f"\n[Trainer] Input shape : {input_shape}")
        print(f"[Trainer] Num classes : {self._data_loader.NumClasses}")
        print(f"[Trainer] Epochs      : {epochs}\n")

        # Build
        self._detector.Build(
            num_classes=self._data_loader.NumClasses,
            input_shape=input_shape,
        )

        # Train
        history = self._detector.Train(train_ds, val_ds, epochs=epochs)

        # Save final weights
        weights_path = self._output_dir / "apple_detector_weights.weights.h5"  # Keras 3 naming convention
        self._detector.Save(weights_path)

        # Evaluate mAP on validation set
        original_val_ds = self._data_loader.GetValidationDataset()
        metrics = self._EvaluateOnValidation(original_val_ds)
        print(f"\n[Trainer] Validation mAP@0.5 = {metrics['mAP']:.4f}")

        return {"history": history.history, "metrics": metrics}

    def _EvaluateOnValidation(
        self,
        val_ds: tf.data.Dataset,
    ) -> Dict[str, Any]:
        """
        Run the detector on every validation batch and compute mAP.

        Returns :
        dict with 'mAP' and 'per_image_AP' keys
        """
        all_predictions:   list = []
        all_ground_truths: list = []

        for images_batch, labels_batch in val_ds:
            for i in range(len(images_batch)):
                image = images_batch[i].numpy()           # (H, W, 3) float32
                gt_boxes_raw  = labels_batch["box_output"][i].numpy()    # (max_boxes, 4)
                gt_labels_raw = labels_batch["class_output"][i].numpy()  # (max_boxes,)

                # Filter padded (zero-label) ground-truth entries
                valid_mask = gt_labels_raw > 0
                gt_list = [
                    BoundingBox(
                        x_min=int(b[1] * image.shape[1]),
                        y_min=int(b[0] * image.shape[0]),
                        x_max=int(b[3] * image.shape[1]),
                        y_max=int(b[2] * image.shape[0]),
                    )
                    for b in gt_boxes_raw[valid_mask]
                ]

                # Run prediction with low threshold to maximise recall for mAP
                # (mAP averages over all thresholds via the P-R curve)
                preds = self._detector.Predict(image, conf_override=0.1)
                pred_list = [
                    BoundingBox(
                        x_min=int(b[1] * image.shape[1]),
                        y_min=int(b[0] * image.shape[0]),
                        x_max=int(b[3] * image.shape[1]),
                        y_max=int(b[2] * image.shape[0]),
                        score=float(s),
                    )
                    for b, s in zip(preds["boxes"], preds["scores"])
                ]

                all_predictions.append(pred_list)
                all_ground_truths.append(gt_list)

        return ComputeMeanAveragePrecision(
            all_predictions,
            all_ground_truths,
            iou_threshold=0.5,
        )
