"""
Text detection using PaddleOCR's DB detector.
Tuned for Vietnamese handwriting: lower thresholds, wider unclip ratio.
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class TextRegion:
    """A detected text region with bounding box and confidence."""
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    crop: np.ndarray | None = None
    text: str = ""
    recognition_confidence: float = 0.0
    method: str = ""


class TextDetector:
    """PaddleOCR DB text detector configured for Vietnamese handwriting."""

    def __init__(
        self,
        det_db_thresh: float = 0.2,
        det_db_box_thresh: float = 0.3,
        det_db_unclip_ratio: float = 2.0,
        det_db_score_mode: str = "slow",
        use_angle_cls: bool = True,
    ):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            det=True,
            rec=False,  # detection only — we use our own recognizers
            cls=use_angle_cls,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            det_db_score_mode=det_db_score_mode,
            show_log=False,
        )

    def detect(self, image: np.ndarray | str) -> list[TextRegion]:
        """Detect text regions in an image.

        Args:
            image: numpy array (H, W, C) or file path

        Returns:
            List of TextRegion objects sorted top-to-bottom
        """
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))

        result = self.ocr.ocr(image, rec=False)

        if result is None or result[0] is None:
            return []

        regions = []
        for box in result[0]:
            # PaddleOCR returns [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            region = TextRegion(
                bbox=box,
                confidence=1.0,  # DB doesn't output per-box confidence directly
            )
            # Crop the region
            region.crop = self._crop_region(image, box)
            regions.append(region)

        # Sort top-to-bottom by y-coordinate of top-left corner
        regions.sort(key=lambda r: r.bbox[0][1])
        return regions

    def _crop_region(self, image: np.ndarray, bbox: list, padding: int = 5) -> np.ndarray:
        """Crop text region with padding."""
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        return image[y:y+h, x:x+w]


# Lazy import to avoid issues when paddleocr not installed
try:
    import cv2
except ImportError:
    pass
