# plate_reader.py — FAST VERSION
# ======================================
# Problem: Previous version ran OCR on 9 candidates × 6 variants = 54 OCR calls
#          → too slow, caused 500 timeout errors.
# Fix:     Max 3 best candidates × 3 variants = 9 OCR calls max.
#          Hard 15-second total timeout.
#          EasyOCR initialized with faster settings.

import cv2
import numpy as np
import re
import logging
import time
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateReader:
    def __init__(self):
        logger.info("Initializing License Plate Reader...")
        try:
            import easyocr
            self.reader = easyocr.Reader(
                ['en'],
                gpu=False,
                verbose=False,
                # Faster detection settings
                detector=True,
                recognizer=True,
            )
            logger.info("✅ EasyOCR ready")
        except Exception as e:
            logger.error(f"EasyOCR init failed: {e}")
            self.reader = None

        self.MAX_CANDIDATES = 3    # only top 3 regions
        self.MAX_VARIANTS   = 3    # only 3 preprocessing variants
        self.TIMEOUT_SEC    = 12   # hard timeout

    # ─────────────────────────────────────────────────────────────────────────
    # PLATE REGION DETECTION  (fast — one pass only)
    # ─────────────────────────────────────────────────────────────────────────
    def _find_plate_region(self, image: np.ndarray):
        """
        Find up to MAX_CANDIDATES plate regions quickly.
        Focuses on the lower half where plates always appear.
        Returns list of (x, y, w, h) sorted by area descending.
        """
        ih, iw = image.shape[:2]
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Search lower 60% of image — plates are never at the top
        roi_y  = int(ih * 0.40)
        roi    = gray[roi_y:, :]

        candidates = []

        # ── Pass 1: Canny + contours ──────────────────────────────────────
        blurred  = cv2.GaussianBlur(roi, (5, 5), 0)
        edges    = cv2.Canny(blurred, 50, 150)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        edges    = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            x, y, w, h = cv2.boundingRect(cnt)
            y += roi_y   # offset to full image coords
            ratio = w / max(h, 1)
            area  = w * h
            if 1.8 < ratio < 6.5 and w > 50 and h > 12 and area > 800:
                candidates.append((x, y, w, h, area))

        # ── Pass 2: Sobel morphology (catches missed plates) ──────────────
        if len(candidates) < 2:
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel  = np.uint8(np.abs(sobelx) / np.abs(sobelx).max() * 255
                              if np.abs(sobelx).max() > 0 else np.zeros_like(roi))
            _, th  = cv2.threshold(sobel, 80, 255, cv2.THRESH_BINARY)
            k      = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
            cnts, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
                x, y, w, h = cv2.boundingRect(cnt)
                y += roi_y
                ratio = w / max(h, 1)
                area  = w * h
                if 1.8 < ratio < 6.5 and w > 50 and h > 12 and area > 800:
                    candidates.append((x, y, w, h, area))

        # Sort by area, return top N as (x,y,w,h)
        candidates.sort(key=lambda c: c[4], reverse=True)
        logger.info(f"Found {len(candidates)} plate candidates (using top {self.MAX_CANDIDATES})")
        return [(x, y, w, h) for x, y, w, h, _ in candidates[:self.MAX_CANDIDATES]]

    # ─────────────────────────────────────────────────────────────────────────
    # PREPROCESSING  (3 fast variants only)
    # ─────────────────────────────────────────────────────────────────────────
    def _preprocess(self, img: np.ndarray):
        """
        Generate 3 fast preprocessing variants.
        All are grayscale for faster EasyOCR processing.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Upscale small plates
        h, w = gray.shape
        if h < 40:
            scale = max(2, 60 // h)
            gray  = cv2.resize(gray, (w * scale, h * scale),
                               interpolation=cv2.INTER_CUBIC)

        variants = []

        # V1: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        variants.append(clahe.apply(gray))

        # V2: Otsu binary
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        # V3: Inverted Otsu (for dark-background plates)
        variants.append(cv2.bitwise_not(otsu))

        return variants

    # ─────────────────────────────────────────────────────────────────────────
    # OCR  (single call, fast settings)
    # ─────────────────────────────────────────────────────────────────────────
    def _ocr(self, img: np.ndarray):
        """Run EasyOCR on one image. Returns list of (text, confidence)."""
        if self.reader is None:
            return []
        try:
            results = self.reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                paragraph=False,
                detail=1,
                batch_size=1,
                workers=0,
            )
            out = []
            for (_, text, conf) in results:
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) >= 2 and conf > 0.15:
                    out.append((clean, conf))
            return out
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────────────────────────────────
    def extract_plate_text(self, image: np.ndarray) -> str:
        """
        Fast pipeline: detect plate region → preprocess → OCR.
        Hard timeout prevents 500 errors.
        Returns plate string or 'UNKNOWN'.
        """
        if image is None or image.size == 0:
            return "UNKNOWN"

        start_time   = time.time()
        all_results  = []   # list of (text, confidence)

        def timed_out():
            return (time.time() - start_time) > self.TIMEOUT_SEC

        # ── Stage 1: Try detected plate regions ──────────────────────────────
        try:
            candidates = self._find_plate_region(image)
        except Exception as e:
            logger.error(f"Region detection error: {e}")
            candidates = []

        for (x, y, w, h) in candidates:
            if timed_out():
                logger.warning("Plate OCR timeout — returning best result so far")
                break

            # Add small padding
            pad = 4
            ih, iw = image.shape[:2]
            x1  = max(0, x - pad);  x2 = min(iw, x + w + pad)
            y1  = max(0, y - pad);  y2 = min(ih, y + h + pad)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            for variant in self._preprocess(crop):
                if timed_out():
                    break
                all_results.extend(self._ocr(variant))

        # ── Stage 2: Bottom-strip fallback ────────────────────────────────────
        if not all_results and not timed_out():
            logger.info("No results from regions — trying bottom strip")
            ih, iw = image.shape[:2]
            bottom = image[int(ih * 0.65):, :]
            for variant in self._preprocess(bottom)[:2]:   # only 2 variants
                if timed_out():
                    break
                all_results.extend(self._ocr(variant))

        # ── Stage 3: Full image OCR ───────────────────────────────────────────
        if not all_results and not timed_out():
            logger.info("Trying full image OCR")
            for variant in self._preprocess(image)[:1]:    # only 1 variant
                all_results.extend(self._ocr(variant))

        elapsed = time.time() - start_time
        logger.info(f"Plate OCR completed in {elapsed:.1f}s")

        # ── Pick best result ──────────────────────────────────────────────────
        valid = [(t, c) for t, c in all_results if len(t) >= 3]
        if not valid:
            logger.warning("No valid plate text found")
            return "UNKNOWN"

        # Weight by confidence; pick most-voted
        votes = {}
        for text, conf in valid:
            votes[text] = votes.get(text, 0) + conf

        best  = max(votes, key=votes.get)
        score = votes[best]
        logger.info(f"✅ Plate: '{best}' (weighted score={score:.2f})")
        return best


# ── Singleton ────────────────────────────────────────────────────────────────
_instance = None

def get_plate_reader():
    global _instance
    if _instance is None:
        _instance = LicensePlateReader()
    return _instance

def extract_plate_text(image: np.ndarray) -> str:
    return get_plate_reader().extract_plate_text(image)