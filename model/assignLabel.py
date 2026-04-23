from __future__ import annotations

from typing import List, Tuple
import numpy as np


def assignLabelsToGrid(
    gt_boxes:   np.ndarray,
    gt_labels:  np.ndarray,
    grid_sizes: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Assign ground-truth boxes to grid cells.

    A cell is positive if any GT box centre falls inside it.
    Returns :
      (total_cells, 5) array:
      col 0   = objectness  (1.0 positive, 0.0 background)
      cols 1-4 = (cy_norm, cx_norm, h_norm, w_norm)  — absolute normalised coords

    Using absolute normalised coords (not offsets) makes the MSE loss simpler
    and more stable than encoding relative cell offsets.

    Parameters :
    gt_boxes  : float32 (max_boxes, 4)  [y_min, x_min, y_max, x_max] in [0,1]
    gt_labels : int32   (max_boxes,)    0=pad, 1=apple
    grid_sizes: [(rows, cols), ...]
    """
    total_cells = sum(r * c for r, c in grid_sizes)
    targets = np.zeros((total_cells, 5), dtype=np.float32)

    valid_boxes = gt_boxes[gt_labels > 0]   # drop padded zeros

    offset = 0
    # Loop through each grid size and assign GT boxes to cells
    for (rows, cols) in grid_sizes:
        for r in range(rows):
            for c in range(cols):
                cell_y1 = r / rows;        cell_y2 = (r + 1) / rows
                cell_x1 = c / cols;        cell_x2 = (c + 1) / cols
                idx = offset + r * cols + c
                # Check if any GT box centre falls inside this cell
                for box in valid_boxes:
                    gy1, gx1, gy2, gx2 = box
                    gt_cy = (gy1 + gy2) / 2
                    gt_cx = (gx1 + gx2) / 2
                    # If the GT box centre is inside the cell, assign it as positive
                    if cell_y1 <= gt_cy < cell_y2 and cell_x1 <= gt_cx < cell_x2:
                        targets[idx, 0] = 1.0                  # positive
                        targets[idx, 1] = gt_cy                # centre y  (absolute)
                        targets[idx, 2] = gt_cx                # centre x  (absolute)
                        targets[idx, 3] = gy2 - gy1            # height
                        targets[idx, 4] = gx2 - gx1            # width
                        break
        offset += rows * cols
    return targets