# detect_production.py
# Helmet violation detection: YOLOv8 (person) + MobileNetV2 CNN (helmet check)
# Updated IMG_SIZE to 96 to match the new transfer learning model.

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

        # ── Load YOLO ────────────────────────────────────────────────────────
        from ultralytics import YOLO
        yolo_path = os.path.join(script_dir, 'yolov8n.pt')
        if not os.path.exists(yolo_path):
            yolo_path = 'yolov8n.pt'
        self.yolo = YOLO(yolo_path)
        logger.info("✅ YOLO loaded")

        # ── Find & Load CNN ──────────────────────────────────────────────────
        model_candidates = [
            os.path.join(script_dir, 'models', 'helmet_cnn_best.keras'),
            os.path.join(script_dir, 'models', 'helmet_cnn.keras'),
            os.path.join(script_dir, 'models', 'helmet_cnn_best.h5'),
            os.path.join(script_dir, 'models', 'helmet_cnn.h5'),
        ]

        self.cnn        = None
        self.img_size   = 96    # must match train_cnn.py IMG_SIZE
        self.using_sigmoid = True

        for path in model_candidates:
            if os.path.exists(path):
                try:
                    from tensorflow.keras.models import load_model
                    self.cnn = load_model(path)
                    out_shape = self.cnn.output_shape
                    self.using_sigmoid = (out_shape[-1] == 1)

                    # Auto-detect input size from model
                    in_shape = self.cnn.input_shape
                    if in_shape[1] is not None:
                        self.img_size = in_shape[1]

                    logger.info(f"✅ CNN loaded: {path}")
                    logger.info(f"   Input size : {self.img_size}×{self.img_size}")
                    logger.info(f"   Output mode: {'sigmoid' if self.using_sigmoid else 'softmax'}")

                    # Smoke test
                    test = np.random.rand(1, self.img_size, self.img_size, 3).astype(np.float32)
                    out  = self.cnn.predict(test, verbose=0)
                    logger.info(f"   Smoke test : {out[0]}")
                    break
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

        if self.cnn is None:
            logger.warning("⚠️  No CNN model — DEMO MODE (all persons = violation)")
            logger.warning("   Run: python train_cnn.py")

        # ── Thresholds ────────────────────────────────────────────────────────
        self.yolo_conf     = 0.40   # YOLO person confidence
        self.head_ratio    = 0.40   # upper 40% of person box = head

        # !! UPDATE THIS after running train_cnn.py !!
        # The script prints: "💡 Set self.helmet_thresh = X.XX"
        # A score >= helmet_thresh means HELMET PRESENT (safe)
        # A score <  helmet_thresh means NO HELMET (violation)
        self.helmet_thresh = 0.50

    # ─────────────────────────────────────────────────────────────────────────
    def _get_score(self, head_roi: np.ndarray) -> float:
        """
        Run CNN on head crop.
        Returns helmet probability: 1.0=definitely helmet, 0.0=no helmet
        Returns -1.0 if CNN not loaded (demo mode).
        """
        if self.cnn is None:
            return -1.0

        try:
            # Resize and convert BGR→RGB (MobileNetV2 trained on RGB)
            img = cv2.resize(head_roi, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)   # MobileNet preprocess built into model

            pred = self.cnn.predict(img[np.newaxis], verbose=0)

            score = float(pred[0][0]) if self.using_sigmoid else float(pred[0][1])
            return score

        except Exception as e:
            logger.error(f"CNN error: {e}")
            return -1.0

    # ─────────────────────────────────────────────────────────────────────────
    def detect_violations(self, frame: np.ndarray,
                          return_visualization: bool = False):
        """
        Detect persons → classify heads → return violations.

        Returns:
            list[dict]  or  (list[dict], annotated_frame)
        """
        annotated     = frame.copy() if return_visualization else None
        violations    = []
        total_persons = 0
        img_h, img_w  = frame.shape[:2]

        try:
            results = self.yolo(frame, classes=[0],
                                conf=self.yolo_conf, verbose=False)
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

                # Crop head region
                hx1 = max(0, x1)
                hx2 = min(img_w, x2)
                hy1 = max(0, y1)
                hy2 = min(img_h, y1 + int((y2 - y1) * self.head_ratio))

                if hy1 >= hy2 or hx1 >= hx2:
                    continue
                head_roi = frame[hy1:hy2, hx1:hx2]
                if head_roi.size == 0:
                    continue

                score = self._get_score(head_roi)

                if score == -1.0:        # demo mode
                    has_helmet = False
                    score      = 0.0
                else:
                    has_helmet = (score >= self.helmet_thresh)

                logger.info(
                    f"Person | score={score:.3f} | thresh={self.helmet_thresh} | "
                    f"{'HELMET ✅' if has_helmet else 'NO HELMET 🚨'}"
                )

                if not has_helmet:
                    violations.append({
                        'bbox':       (x1, y1, x2, y2),
                        'confidence': round(float(det_conf), 3),
                        'score':      round(score, 3),
                        'timestamp':  datetime.now().isoformat(),
                    })

                if return_visualization:
                    color = (0, 200, 0) if has_helmet else (0, 0, 220)
                    label = (f"Helmet {score:.0%}"
                             if has_helmet else f"NO HELMET {score:.0%}")
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated,
                                  (x1, max(y1 - th - 12, 0)),
                                  (x1 + tw + 8, y1), color, -1)
                    cv2.putText(annotated, label,
                                (x1 + 4, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2, cv2.LINE_AA)

        logger.info(f"Persons: {total_persons} | Violations: {len(violations)}")

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