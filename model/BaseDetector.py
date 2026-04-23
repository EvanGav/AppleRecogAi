from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ── Abstract interface (DIP) ──────────────────────────────────────────────────

class BaseDetector(abc.ABC):
    """Thin abstract interface"""

    @abc.abstractmethod
    def Build(self, num_classes: int, input_shape: Tuple[int, int, int]) -> None:
        """Build and initialise the model."""

    @abc.abstractmethod
    def Train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset:   tf.data.Dataset,
        epochs: int,
    ) -> keras.callbacks.History:
        """Train and return history."""

    @abc.abstractmethod
    def Predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference. image: float32 (H,W,3). Returns boxes/scores/labels."""

    @abc.abstractmethod
    def Save(self, path: Path) -> None:
        """Save weights."""

    @abc.abstractmethod
    def Load(self, path: Path) -> None:
        """Load weights."""