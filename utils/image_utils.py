from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.BoundingBox import BoundingBox


DEFAULT_TARGET_SIZE: Tuple[int, int] = (320, 320)  # (width, height) for the model
BOX_COLOR: Tuple[int, int, int]     = (0, 200, 80)   # bright green in BGR
BOX_THICKNESS: int                  = 2
FONT                                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float                   = 0.55
FONT_THICKNESS: int                 = 1



def LoadImageAsArray(image_path: Path) -> np.ndarray:
    """
    Load an image from disk and return it as a uint8 RGB numpy array.

    Parameters :
    image_path : Path

    Returns :
    np.ndarray  shape (H, W, 3), dtype uint8, channels = RGB
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def ResizeImage(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    Resize image to target_size using bilinear interpolation.

    Parameters :
    image : np.ndarray  (H, W, 3) uint8
    target_size : (width, height)

    Returns :
    np.ndarray  (target_height, target_width, 3) uint8
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def NormaliseImage(image: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 image to float32 with values in [0, 1].

    Parameters :
    image : np.ndarray  uint8

    Returns :
    np.ndarray  float32
    """
    return image.astype(np.float32) / 255.0


def PrepareModelInput(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    Full preprocessing pipeline: resize -> normalise -> add batch dimension.

    Returns :
    np.ndarray  shape (1, H, W, 3) float32  — ready to pass to model.predict()
    """
    resized     = ResizeImage(image, target_size)
    normalised  = NormaliseImage(resized)
    batched     = np.expand_dims(normalised, axis=0)
    return batched


def DrawBoundingBoxes(
    image: np.ndarray,
    boxes: List[BoundingBox],
    *,
    show_score: bool = True,
) -> np.ndarray:
    """
    Draw detection boxes on image and return the result.

    Parameters :
    image      : np.ndarray  uint8 RGB
    boxes      : list of BoundingBox (pixel coordinates)
    show_score : bool  — whether to render the confidence score label

    Returns :
    np.ndarray  uint8 RGB with boxes drawn
    """
    output = image.copy()

    for box in boxes:
        # Convert RGB → BGR for OpenCV drawing, then back
        frame_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Draw rectangle
        cv2.rectangle(
            frame_bgr,
            (box.x_min, box.y_min),
            (box.x_max, box.y_max),
            BOX_COLOR,
            BOX_THICKNESS,
        )

        # Render label + score above the box
        label_text = f"{box.label}"
        if show_score:
            label_text += f"  {box.score:.0%}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, FONT, FONT_SCALE, FONT_THICKNESS
        )
        label_y = max(box.y_min - 4, text_h + baseline)

        # Background rectangle for text readability
        cv2.rectangle(
            frame_bgr,
            (box.x_min, label_y - text_h - baseline),
            (box.x_min + text_w, label_y + baseline),
            BOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            frame_bgr,
            label_text,
            (box.x_min, label_y),
            FONT,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

        output = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return output


def ImageArrayToBase64(image: np.ndarray, fmt: str = "JPEG") -> str:
    """
    Encode a uint8 RGB numpy array as a base-64 string for embedding in HTML.

    Parameters :
    image : np.ndarray  uint8 RGB
    fmt   : str  Pillow format name, e.g. 'JPEG' or 'PNG'

    Returns :
    str  — data URI: "data:image/jpeg;base64,..."
    """
    pil_image = Image.fromarray(image)
    buffer    = io.BytesIO()
    pil_image.save(buffer, format=fmt, quality=90)
    encoded   = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime      = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{encoded}"
