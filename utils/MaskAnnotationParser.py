import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
from utils.BoundingBox import BoundingBox

class MaskAnnotationParser:
    """
    Parses a MinneApple segmentation mask and extracts per-apple bounding boxes.
    """

    def __init__(self, min_area: int = 50) -> None:
        """
        Parameters :
        min_area : int
            Connected components smaller than this (in pixels²) are ignored.
            Helps discard mask noise.
        """
        self._min_area = min_area

    def ParseMask(self, mask_path: Path) -> List[BoundingBox]:
        """
        Read a binary PNG mask and return one BoundingBox per apple instance.

        Parameters :
        mask_path : Path
            Path to the .png mask file (white pixels = apple, black = background).

        Returns :
        List[BoundingBox]
            Zero or more bounding boxes, one per detected apple region.
        """
        # Load mask as greyscale; each white region = one apple
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot open mask: {mask_path}")

        # Binarise: MinneApple masks use pixel value = instance id (1, 2, 3...)
        # Threshold must be 0 otherwise all apple pixels are zeroed out
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        # Label connected components (each = one apple instance)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        boxes: List[BoundingBox] = []
        # Label 0 is the background — skip it
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self._min_area:
                continue  # discard noise blobs

            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]

            boxes.append(BoundingBox(
                x_min = x,
                y_min = y,
                x_max = x + w,
                y_max = y + h,
            ))

        return boxes
