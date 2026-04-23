import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class BoundingBox:
    """
    Representation of a single detection box.
    Coordinates are in pixel space: (x_min, y_min, x_max, y_max).
    """
    x_min: int
    y_min: int
    x_max:  int
    y_max:  int
    label:  str = "apple"
    score:  float = 1.0  # confidence (1.0 for ground-truth, model score for predictions)

    def ToNormalized(self, image_width: int, image_height: int) -> "BoundingBox":
        """Return a copy with coordinates normalised"""
        return BoundingBox(
            x_min  = self.x_min / image_width,
            y_min  = self.y_min / image_height,
            x_max  = self.x_max / image_width,
            y_max  = self.y_max / image_height,
            label  = self.label,
            score  = self.score,
        )

    def ToList(self) -> List[float]:
        """Return [x_min, y_min, x_max, y_max] as a plain list."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]