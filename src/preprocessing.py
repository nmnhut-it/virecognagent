"""
Image preprocessing for Vietnamese handwritten text.
Higher resolution (192px) than standard OCR (48-128px) to preserve diacritics.
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(
    image: np.ndarray | Image.Image,
    target_height: int = 192,
    clahe_clip: float = 2.0,
    clahe_grid: tuple = (8, 8),
    deskew: bool = True,
    denoise: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for Vietnamese handwriting."""

    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Denoise (preserve edges for diacritics)
    if denoise:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # CLAHE for contrast enhancement (critical for faded diacritics)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray = clahe.apply(gray)

    # Deskew
    if deskew:
        gray = _deskew(gray)

    # Binarize (adaptive threshold preserves thin diacritic strokes)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    # Resize maintaining aspect ratio (higher than default for diacritics)
    if target_height > 0:
        h, w = binary.shape
        scale = target_height / h
        new_w = int(w * scale)
        binary = cv2.resize(binary, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    return binary


def _deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Correct skew in handwritten text."""
    try:
        from deskew import determine_skew
        angle = determine_skew(image)
        if angle is None or abs(angle) > max_angle:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    except ImportError:
        # Fallback: simple Hough-based deskew
        coords = np.column_stack(np.where(image < 128))
        if len(coords) < 100:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > max_angle:
            return image
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_text_region(image: np.ndarray, bbox: list[list[int]], padding: int = 5) -> np.ndarray:
    """Crop a text region from image given quadrilateral bbox."""
    pts = np.array(bbox, dtype=np.float32)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    crop = image[y:y+h, x:x+w]
    return crop
