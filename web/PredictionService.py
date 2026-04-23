from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image

from model.SSDMobileNetDetector import BaseDetector
from utils.image_utils   import (
    PrepareModelInput,
    DrawBoundingBoxes,
    ImageArrayToBase64,
    DEFAULT_TARGET_SIZE,
)
from utils.BoundingBox import BoundingBox


class PredictionService:
    """
    Wraps the detector and exposes a single public method: RunOnPilImage.

    bridge between the HTTP layer (PIL images, base64 results)
    and the ML layer (numpy arrays, BoundingBox objects).
    """

    def __init__(self, detector: BaseDetector, weights_path: Path) -> None:
        self._detector      = detector
        self._model_loaded  = False
        self._weights_path  = weights_path
        self._TryLoadModel()

    def _TryLoadModel(self) -> None:
        """Attempt to build and load weights"""
        try:
            self._detector.Build(
                num_classes=2,
                input_shape=(DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0], 3),
            )
            if self._weights_path.exists():
                self._detector.Load(self._weights_path)
                self._model_loaded = True
                print(f"[PredictionService] Model loaded from {self._weights_path}")
            else:
                print(
                    f"[PredictionService] No weights at {self._weights_path}. "
                    "Running in DEMO mode (random predictions)."
                )
        except Exception as exc:
            print(f"[PredictionService] Model init error: {exc}")

    @property
    def IsModelLoaded(self) -> bool:
        return self._model_loaded

    def RunOnPilImage(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Run detection on a PIL image and return a result dict.

        Parameters :
        pil_image : PIL.Image

        Returns :
        dict:
            'annotated_image' : base64 data URI of the image with boxes drawn
            'detections'      : list of {'x_min', 'y_min', 'x_max', 'y_max', 'score'}
            'count'           : number of apples detected
            'model_loaded'    : bool
        """
        # Convert PIL → RGB numpy
        rgb = np.array(pil_image.convert("RGB"))
        orig_h, orig_w = rgb.shape[:2]

        if self._model_loaded:
            boxes_info = self._RunRealModel(rgb, orig_w, orig_h)
        else:
            boxes_info = self._RunDemoMode(orig_w, orig_h)

        # Draw boxes on original (full-resolution) image
        annotated = DrawBoundingBoxes(rgb, boxes_info)
        img_b64   = ImageArrayToBase64(annotated, fmt="JPEG")

        detections = [
            {
                "x_min": box.x_min, "y_min": box.y_min,
                "x_max": box.x_max, "y_max": box.y_max,
                "score": round(box.score, 3),
                "label": box.label,
            }
            for box in boxes_info
        ]

        return {
            "annotated_image": img_b64,
            "detections":      detections,
            "count":           len(detections),
            "model_loaded":    self._model_loaded,
        }

    def _RunRealModel(
        self,
        rgb: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> List[BoundingBox]:
        """Run the real TF model and map normalised coords back to pixel space.
        """
        prepared = PrepareModelInput(rgb)[0]   # (H, W, 3) float32
        preds    = self._detector.Predict(prepared)

        # Scale factor to compensate for domain mismatch (orchard vs close-up)
        # Set to 1.0 to disable, 1.5-2.0 for close-up images
        BOX_SCALE = float(os.getenv("BOX_SCALE", "3.5"))

        boxes: List[BoundingBox] = []
        for box_norm, score in zip(preds["boxes"], preds["scores"]):
            y_min, x_min, y_max, x_max = box_norm

            # Scale box around its centre
            cy = (y_min + y_max) / 2
            cx = (x_min + x_max) / 2
            h  = (y_max - y_min) * BOX_SCALE
            w  = (x_max - x_min) * BOX_SCALE
            y_min = max(0.0, cy - h / 2)
            y_max = min(1.0, cy + h / 2)
            x_min = max(0.0, cx - w / 2)
            x_max = min(1.0, cx + w / 2)

            boxes.append(BoundingBox(
                x_min = int(x_min * orig_w),
                y_min = int(y_min * orig_h),
                x_max = int(x_max * orig_w),
                y_max = int(y_max * orig_h),
                score = float(score),
            ))
        return boxes

    def _RunDemoMode(self, orig_w: int, orig_h: int) -> List[BoundingBox]:
        """
        Return a few plausible fake boxes so the UI is usable before training.
        In production this path is never taken.
        """
        rng   = np.random.default_rng(42)
        boxes = []
        for _ in range(rng.integers(2, 6)):
            cx = rng.integers(orig_w // 5, orig_w * 4 // 5)
            cy = rng.integers(orig_h // 5, orig_h * 4 // 5)
            r  = rng.integers(30, min(orig_w, orig_h) // 5)
            boxes.append(BoundingBox(
                x_min = max(0, cx - r),
                y_min = max(0, cy - r),
                x_max = min(orig_w, cx + r),
                y_max = min(orig_h, cy + r),
                score = round(float(rng.uniform(0.4, 0.95)), 3),
            ))
        return boxes

