from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.MaskAnnotationParser import MaskAnnotationParser, BoundingBox
from utils.image_utils       import LoadImageAsArray, PrepareModelInput, DEFAULT_TARGET_SIZE
from data.BaseDataLoader     import BaseDataLoader


class MinneAppleDataLoader(BaseDataLoader):
    """
    Loads the MinneApple detection dataset from disk.

    Each mask is a binary image where connected white blobs = apple instances.
    We convert those blobs to bounding boxes via MaskAnnotationParser.
    """

    # MinneApple has a single class: apple (background = 0, apple = 1)
    _NUM_CLASSES: int = 2

    def __init__(
        self,
        data_dir: Path,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        batch_size: int = 4,
        max_boxes: int = 50,
        validation_split: float = 0.20,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        data_dir         : root folder of the extracted dataset
        target_size      : (width, height) to resize images to
        batch_size       : mini-batch size
        max_boxes        : pad/clip all annotation tensors to this length
        validation_split : fraction of train set used for validation (20% = ~134 images)
        seed             : random seed for the split
        """
        self._data_dir         = Path(data_dir)
        self._target_size      = target_size
        self._batch_size       = batch_size
        self._max_boxes        = max_boxes
        self._validation_split = validation_split
        self._rng              = np.random.default_rng(seed)
        self._parser           = MaskAnnotationParser(min_area=50)

        # Collect file paths eagerly so we can split deterministically
        self._train_pairs, self._val_pairs = self._SplitTrainVal()

    # ── BaseDataLoader interface ───────────────────────────────────────────────

    def GetTrainDataset(self) -> tf.data.Dataset:
        """Return a shuffled, batched training dataset."""
        return self._BuildDataset(self._train_pairs, shuffle=True)

    def GetValidationDataset(self) -> tf.data.Dataset:
        """Return a batched validation dataset (no shuffle)."""
        return self._BuildDataset(self._val_pairs, shuffle=False)

    @property
    def NumClasses(self) -> int:
        return self._NUM_CLASSES

    # ── Private helpers ────────────────────────────────────────────────────────

    def _CollectPairs(self, split: str) -> List[Tuple[Path, Path]]:
        """
        Gather (image_path, mask_path) pairs for a given split folder.
        Skips images that have no corresponding mask.
        """
        image_dir = self._data_dir / split / "images"
        mask_dir  = self._data_dir / split / "masks"

        pairs: List[Tuple[Path, Path]] = []
        for img_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def _SplitTrainVal(
        self,
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """
        Split the 'train' folder into train / val subsets.
        Returns (train_pairs, val_pairs).
        """
        all_pairs = self._CollectPairs("train")
        indices   = self._rng.permutation(len(all_pairs))
        n_val     = max(1, int(len(all_pairs) * self._validation_split))
        val_idx   = set(indices[:n_val].tolist())

        train_pairs = [p for i, p in enumerate(all_pairs) if i not in val_idx]
        val_pairs   = [p for i, p in enumerate(all_pairs) if i in val_idx]
        return train_pairs, val_pairs

    def _LoadSingleSample(
        self,
        image_path: Path,
        mask_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load one (image, boxes, labels) sample.

        Returns :
            image  : float32 (H, W, 3) — normalised
            boxes  : float32 (max_boxes, 4) — normalised [y1,x1,y2,x2] as TF expects
            labels : int32   (max_boxes,)   — class indices (1 = apple, 0 = pad)
        """
        image = LoadImageAsArray(image_path)
        h, w  = image.shape[:2]

        # Preprocess image (resize + normalise), drop batch dim
        prepared = PrepareModelInput(image, self._target_size)[0]

        # Parse mask → boxes (pixel coords) → normalised [y1,x1,y2,x2]
        raw_boxes: List[BoundingBox] = self._parser.ParseMask(mask_path)

        boxes_list  = []
        labels_list = []
        for box in raw_boxes[: self._max_boxes]:
            # Convert to normalised [y1, x1, y2, x2] format expected by TF and model
            boxes_list.append([
                box.y_min / h,
                box.x_min / w,
                box.y_max / h,
                box.x_max / w,
            ])
            labels_list.append(1)  

        # Pad to max_boxes with zeros
        n = len(boxes_list)
        pad = self._max_boxes - n
        boxes_arr  = np.array(boxes_list  + [[0, 0, 0, 0]] * pad, dtype=np.float32)
        labels_arr = np.array(labels_list + [0]            * pad, dtype=np.int32)

        return prepared, boxes_arr, labels_arr

    def _BuildDataset(
        self,
        pairs: List[Tuple[Path, Path]],
        shuffle: bool,
    ) -> tf.data.Dataset:
        """
        Build a tf.data.Dataset from a list of (image_path, mask_path) pairs.
        """
        # Pre-load everything into numpy arrays (dataset is small enough)
        images_list, boxes_list, labels_list = [], [], []

        print(f"Loading {len(pairs)} samples…")
        # We use tqdm here for a progress bar since loading can be slow, and it's nice to see progress
        for img_path, mask_path in tqdm(pairs):
            try:
                img, boxes, labels = self._LoadSingleSample(img_path, mask_path)
                images_list.append(img)
                boxes_list.append(boxes)
                labels_list.append(labels)
            except Exception as exc:
                print(f"  [WARN] Skipping {img_path.name}: {exc}")

        images = np.stack(images_list)
        boxes  = np.stack(boxes_list)
        labels = np.stack(labels_list)

        # Keys must match the model's named output layers exactly:
        # "box_output" -> regression head, "class_output" -> classification head
        ds = tf.data.Dataset.from_tensor_slices((images, {"box_output": boxes, "class_output": labels}))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

        ds = ds.batch(self._batch_size)

        # Apply augmentation only during training (shuffle=True means train set)
        if shuffle:
            ds = ds.map(self._Augment, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _Augment(
        self,
        images: tf.Tensor,
        labels: dict,
    ) -> tuple:
        """
        Augmentation applied per-batch during training.

        Includes horizontal flip with box mirroring:
          flipped_x = 1 - x  for both x_min and x_max.
        Boxes are stored as [y_min, x_min, y_max, x_max] so we mirror
        indices 1 and 3.
        """
        boxes = labels["box_output"]    # (B, 50, 4)  [y1, x1, y2, x2]

        # Per-image horizontal flip (independent 50% per image) 
        B = tf.shape(images)[0]
        flip_mask = tf.random.uniform([B]) > 0.5   # (B,) bool per image

        # Flip images where mask is True
        images = tf.where(
            tf.reshape(flip_mask, [B, 1, 1, 1]),
            tf.image.flip_left_right(images),
            images,
        )

        # Mirror boxes where mask is True — boxes shape (B, 50, 4)
        y1 = boxes[..., 0:1]
        x1_flipped = 1.0 - boxes[..., 3:4]   # new x_min = 1 - old x_max
        y2 = boxes[..., 2:3]
        x2_flipped = 1.0 - boxes[..., 1:2]   # new x_max = 1 - old x_min
        flipped_boxes = tf.concat([y1, x1_flipped, y2, x2_flipped], axis=-1)
        flip_mask_boxes = tf.reshape(flip_mask, [B, 1, 1])
        boxes = tf.where(flip_mask_boxes, flipped_boxes, boxes)

        # Color jitter 
        images = tf.image.random_brightness(images, max_delta=0.2)
        images = tf.image.random_contrast(images, lower=0.75, upper=1.25)
        images = tf.image.random_saturation(images, lower=0.6, upper=1.4)
        images = tf.image.random_hue(images, max_delta=0.08)
        images = tf.clip_by_value(images, 0.0, 1.0)

        labels = dict(labels)
        labels["box_output"] = boxes
        return images, labels
