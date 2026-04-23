from __future__ import annotations

import abc
import tensorflow as tf

from utils.image_utils       import LoadImageAsArray, PrepareModelInput, DEFAULT_TARGET_SIZE

class BaseDataLoader(abc.ABC):
    """
    Minimal interface every data loader must satisfy.
    """

    @abc.abstractmethod
    def GetTrainDataset(self) -> tf.data.Dataset:
        """Return a batched tf.data.Dataset for training."""

    @abc.abstractmethod
    def GetValidationDataset(self) -> tf.data.Dataset:
        """Return a batched tf.data.Dataset for validation."""

    @property
    @abc.abstractmethod
    def NumClasses(self) -> int:
        """Number of distinct object classes (including background at index 0)."""