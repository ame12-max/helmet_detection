# detect_production.py  ← THE ONLY DETECTION FILE YOU NEED
# ============================================================
# Helmet violation detection:
#   1. YOLOv8 detects persons in the frame
#   2. Keras CNN checks each person's head for a helmet
# ============================================================

import cv2
import numpy as np
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelmetDetector:
    def __init__(self):
        logger.info("Initializing Helmet Detector...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # ── 1. Load YOLO ────────────────────────────────────────────────────
        from ultralytics import YOLO
        yolo_path = os.path.join(script_dir, 'yolov8n.pt')
        if not os.path.exists(yolo_path):
            yolo_path = 'yolov8n.pt'
        self.yolo = YOLO(yolo_path)
        logger.info("✅ YOLO loaded")

        # ── 2. Find & Load CNN ──────────────────────────────────────────────
        model_candidates = [
            os.path.join(script_dir, 'models', 'helmet_cnn_best.keras'),
            os.path.join(script_dir, 'models', 'helmet_cnn.keras'),
            os.path.join(script_dir, 'models', 'helmet_cnn_best.h5'),
            os.path.join(script_dir, 'models', 'helmet_cnn.h5'),
        ]

        self.cnn = None
        self.using_sigmoid = True

        for path in model_candidates:
            if os.path.exists(path):
                try:
                    from tensorflow.keras.models import load_model
                    self.cnn = load_model(path)
                    self.using_sigmoid = (self.cnn.output_shape[-1] == 1)
                    logger.info(f"✅ CNN loaded: {path}")
                    logger.info(f"   Mode: {'sigmoid' if self.using_sigmoid else 'softmax'}")
                    # Smoke test
                    test = np.random.rand(1, 64, 64, 3).astype(np.float32)
                    out = self.cnn.predict(test, verbose=0)
                    logger.info(f"   Smoke test: {out[0]}")
                    break
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        if self.cnn is None:
            logger.warning("⚠️  No CNN model found — DEMO MODE (all persons = violation)")
            logger.warning("   Run: python train_cnn.py  to train the model first")

        # Detection thresholds
        self.yolo_conf     = 0.40  # YOLO minimum person confidence
        self.head_ratio    = 0.40  # upper 40% of person box = head
        self.helmet_thresh = 0.50  # CNN score >= this means helmet present

    # ─────────────────────────────────────────────────────────────────────────
    def _has_helmet(self, head_roi: np.ndarray):
        """
        Classify a cropped head ROI.
        Returns: (has_helmet: bool, score: float)
        score = probability helmet IS present (1.0 = definitely helmet)
        """
        if self.cnn is None:
            return False, 0.0   # demo mode: always a violation

        try:
            img = cv2.resize(head_roi, (64, 64)).astype(np.float32) / 255.0
            pred = self.cnn.predict(img[np.newaxis], verbose=0)

            if self.using_sigmoid:
                # train_cnn.py: Dense(1, sigmoid)
                # 1.0 = with_helmet, 0.0 = without_helmet
                score = float(pred[0][0])
            else:
                # Legacy softmax: [without_helmet_prob, with_helmet_prob]
                score = float(pred[0][1])

            return (score >= self.helmet_thresh), score

        except Exception as e:
            logger.error(f"CNN error: {e}")
            return False, 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def detect_violations(self, frame: np.ndarray, return_visualization: bool = False):
        """
        Full pipeline: detect persons → classify heads → return violations.

        Args:
            frame               : OpenCV BGR image
            return_visualization: if True returns (violations, annotated_frame)
                                  if False returns violations only

        Returns:
            violations list — each item: {bbox, confidence, score, timestamp}
        """
        annotated      = frame.copy() if return_visualization else None
        violations     = []
        total_persons  = 0
        img_h, img_w   = frame.shape[:2]

        try:
            results = self.yolo(frame, classes=[0], conf=self.yolo_conf, verbose=False)
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return (violations, annotated) if return_visualization else violations

        for r in results:
            if r.boxes is None:
                continue

            boxes          = r.boxes.xyxy.cpu().numpy()
            confs          = r.boxes.conf.cpu().numpy()
            total_persons += len(boxes)

            for box, det_conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)

                # Crop head region (upper head_ratio of person bbox)
                hx1 = max(0, x1)
                hx2 = min(img_w, x2)
                hy1 = max(0, y1)
                hy2 = min(img_h, y1 + int((y2 - y1) * self.head_ratio))

                if hy1 >= hy2 or hx1 >= hx2:
                    continue
                head_roi = frame[hy1:hy2, hx1:hx2]
                if head_roi.size == 0:
                    continue

                has_helmet, score = self._has_helmet(head_roi)

                if not has_helmet:
                    violations.append({
                        'bbox':       (x1, y1, x2, y2),
                        'confidence': round(float(det_conf), 3),
                        'score':      round(score, 3),
                        'timestamp':  datetime.now().isoformat(),
                    })

                # Draw bounding box on annotated frame
                if return_visualization:
                    color = (0, 200, 0) if has_helmet else (0, 0, 220)
                    label = f"Helmet {score:.0%}" if has_helmet else f"NO HELMET {score:.0%}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(annotated,
                                  (x1, max(y1 - th - 10, 0)),
                                  (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated, label,
                                (x1 + 3, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2, cv2.LINE_AA)

        logger.info(f"Persons: {total_persons}  |  Violations: {len(violations)}")

        if return_visualization:
            return violations, annotated
        return violations


# ── Singleton ────────────────────────────────────────────────────────────────
_instance = None

def get_detector():
    global _instance
    if _instance is None:
        _instance = HelmetDetector()
    return _instance