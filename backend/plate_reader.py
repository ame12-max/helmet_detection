# plate_reader.py  ← THE ONLY PLATE READER FILE YOU NEED
# =========================================================
# License plate detection (OpenCV) + OCR (EasyOCR)
# =========================================================

import cv2
import numpy as np
import re
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateReader:
    def __init__(self):
        logger.info("Initializing License Plate Reader...")
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("✅ EasyOCR ready")
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            self.reader = None

    # ─────────────────────────────────────────────────────────────────────────
    def _find_plate_region(self, image: np.ndarray):
        """
        Locate the license plate bounding box via edge detection + contours.
        Returns (x, y, w, h) or None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Method 1 – Canny edges → rectangular contours
        for ksize in [(3, 3), (5, 5)]:
            blurred  = cv2.GaussianBlur(gray, ksize, 0)
            edges    = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
                area = cv2.contourArea(cnt)
                if not (500 < area < 50000):
                    continue
                peri   = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if 4 <= len(approx) <= 8:
                    x, y, w, h = cv2.boundingRect(cnt)
                    ratio = w / max(h, 1)
                    if 1.5 < ratio < 5.5 and w > 50 and h > 15:
                        logger.info(f"Plate region: ({x},{y}) {w}×{h} ratio={ratio:.2f}")
                        return x, y, w, h

        # Method 2 – Adaptive threshold on scaled image
        h_img, w_img = gray.shape
        for scale in [0.75, 1.0]:
            small  = cv2.resize(gray, (int(w_img * scale), int(h_img * scale)))
            thresh = cv2.adaptiveThreshold(small, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 200:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 40 and h > 15 and w > h * 1.5:
                    return (int(x / scale), int(y / scale),
                            int(w / scale), int(h / scale))

        return None

    # ─────────────────────────────────────────────────────────────────────────
    def _ocr(self, img: np.ndarray):
        """Run EasyOCR and return a list of cleaned alphanumeric strings."""
        if self.reader is None:
            return []
        try:
            results = self.reader.readtext(
                img,
                paragraph=True,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=0
            )
            return [re.sub(r'[^A-Z0-9]', '', str(r).upper())
                    for r in results]
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────────────────────
    def extract_plate_text(self, image: np.ndarray) -> str:
        """
        Main entry point.
        Returns the plate number string, or 'UNKNOWN' if not found.
        """
        try:
            region = self._find_plate_region(image)
            if region is None:
                logger.warning("No plate region found")
                return "UNKNOWN"

            x, y, w, h = region
            plate_img = image[y:y + h, x:x + w]
            if plate_img.size == 0:
                return "UNKNOWN"

            results = []

            # Pass 1 – original crop
            results.extend(self._ocr(plate_img))

            # Pass 2 – 2× upscale + CLAHE contrast enhancement
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) \
                   if len(plate_img.shape) == 3 else plate_img.copy()
            up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(up)
            results.extend(self._ocr(enhanced))

            # Pass 3 – Otsu binarisation
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results.extend(self._ocr(binary))

            # Pick the most frequent result with >= 4 characters
            valid = [r for r in results if len(r) >= 4]
            if valid:
                plate = Counter(valid).most_common(1)[0][0]
                logger.info(f"✅ Plate: {plate}")
                return plate

            logger.warning("OCR produced no valid plate text")
            return "UNKNOWN"

        except Exception as e:
            logger.error(f"Plate extraction error: {e}")
            return "UNKNOWN"


# ── Singleton ────────────────────────────────────────────────────────────────
_instance = None

def get_plate_reader():
    global _instance
    if _instance is None:
        _instance = LicensePlateReader()
    return _instance

def extract_plate_text(image: np.ndarray) -> str:
    return get_plate_reader().extract_plate_text(image)