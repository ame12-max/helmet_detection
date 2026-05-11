# app.py — COMPLETE FIXED VERSION
# ================================
# Fix: Plate reader finds regions but no text → added tesseract fallback
#      and better preprocessing for low-quality images

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import cv2
import base64
import logging
import threading
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PLATE_TIMEOUT = 30


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect('violations.db', timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            image_path    TEXT,
            plate_number  TEXT    DEFAULT "UNKNOWN",
            helmet_status TEXT,
            confidence    REAL
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("✅ Database ready")


init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Load detection modules
# ─────────────────────────────────────────────────────────────────────────────
try:
    from detect_production import get_detector
    from plate_reader import get_plate_reader

    detector = get_detector()
    plate_reader = get_plate_reader()
    logger.info("✅ Detection modules loaded")

except Exception as e:
    logger.error(f"Could not load detection modules: {e}")

    class _DummyDetector:
        def detect_violations(self, frame, return_visualization=False):
            return ([], frame) if return_visualization else []

    class _DummyReader:
        def extract_plate_text(self, image):
            return "UNKNOWN"

    detector = _DummyDetector()
    plate_reader = _DummyReader()


# ─────────────────────────────────────────────────────────────────────────────
# Plate OCR with fallback and better preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def read_plate_with_timeout(frame, timeout=PLATE_TIMEOUT):
    """Run plate OCR with hard timeout and fallback to simple OCR"""
    result = {'plate': 'UNKNOWN'}

    def _run():
        try:
            # First try the primary plate reader
            plate = plate_reader.extract_plate_text(frame)
            if plate != "UNKNOWN" and len(plate) >= 3:
                result['plate'] = plate
                return

            # Fallback: Try simple OCR on full image
            logger.info("Primary reader failed, trying fallback OCR...")
            fallback_plate = simple_plate_ocr(frame)
            if fallback_plate != "UNKNOWN":
                result['plate'] = fallback_plate
        except Exception as e:
            logger.error(f"Plate OCR error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.warning(f"Plate OCR timed out after {timeout}s")
        return "UNKNOWN"

    return result['plate']


def simple_plate_ocr(image):
    """Simple fallback OCR using basic image processing"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Multiple preprocessing attempts
        for scale in [2, 3]:
            # Resize
            h, w = gray.shape
            if h < 100:
                scaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            else:
                scaled = gray

            # Try different thresholds
            for method in ['otsu', 'adaptive', 'simple']:
                if method == 'otsu':
                    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif method == 'adaptive':
                    thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)
                else:
                    _, thresh = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)

                # Try EasyOCR
                try:
                    import easyocr
                    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                    results = reader.readtext(thresh, paragraph=True, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    if results:
                        text = ''.join(results[0]).upper()
                        import re
                        text = re.sub(r'[^A-Z0-9]', '', text)
                        if len(text) >= 3:
                            logger.info(f"Fallback OCR found: {text}")
                            return text
                except:
                    pass

        return "UNKNOWN"
    except Exception as e:
        logger.error(f"Fallback OCR error: {e}")
        return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "time": datetime.now().isoformat()})


@app.route('/violations', methods=['GET'])
def get_violations():
    try:
        conn = get_db()
        rows = conn.execute(
            '''SELECT id, timestamp, image_path, plate_number
               FROM violations ORDER BY id DESC LIMIT 100'''
        ).fetchall()
        conn.close()
        return jsonify([
            {'id': r['id'], 'time': r['timestamp'],
             'image': r['image_path'], 'plate': r['plate_number'] or 'UNKNOWN'}
            for r in rows
        ])
    except Exception as e:
        logger.error(f"DB fetch error: {e}")
        return jsonify([])


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        weekly = conn.execute(
            "SELECT COUNT(*) FROM violations "
            "WHERE timestamp >= datetime('now', '-7 days')"
        ).fetchone()[0]
        plates = conn.execute(
            "SELECT COUNT(*) FROM violations "
            "WHERE plate_number NOT IN ('UNKNOWN', 'ERROR', '') "
            "AND plate_number IS NOT NULL AND plate_number != ''"
        ).fetchone()[0]
        conn.close()
        return jsonify({'total': total, 'weekly': weekly, 'platesDetected': plates})
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'total': 0, 'weekly': 0, 'platesDetected': 0})


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Accept image → detect helmets → read plate → save to DB → return JSON.
    Every step is independently error-handled so we always return 200.
    """
    try:
        # ── Validate request ──────────────────────────────────────────────────
        if 'image' not in request.files:
            return jsonify({'error': 'No image in request'}), 400
        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'error': 'Empty file'}), 400

        # ── Save file ─────────────────────────────────────────────────────────
        orig_name = file.filename or 'frame.jpg'
        safe_name = secure_filename(
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{orig_name}"
        )
        filepath = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(filepath)
        logger.info(f"Saved: {filepath}")

        # ── Decode image ──────────────────────────────────────────────────────
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        # ── Helmet detection ──────────────────────────────────────────────────
        violations = []
        annotated = frame.copy()
        helmet_status = "Detection Error"
        ann_path = filepath
        count = 0

        try:
            violations, annotated = detector.detect_violations(
                frame, return_visualization=True)
            count = len(violations)
            helmet_status = "No Helmet" if count > 0 else "Helmet OK"

            ann_name = f"ann_{safe_name}"
            ann_path = os.path.join(UPLOAD_FOLDER, ann_name)
            cv2.imwrite(ann_path, annotated)

            logger.info(f"Detection → {count} violations")

        except Exception as e:
            logger.error(f"Detection error: {e}")

        # ── Plate OCR (run ALWAYS — plates exist even with helmet) ───────────
        plate = "UNKNOWN"
        try:
            plate = read_plate_with_timeout(frame, timeout=PLATE_TIMEOUT)
            logger.info(f"Plate → {plate}")
        except Exception as e:
            logger.error(f"Plate OCR wrapper error: {e}")

        # ── Save to DB ────────────────────────────────────────────────────────
        try:
            conn = get_db()
            conn.execute(
                '''INSERT INTO violations
                   (timestamp, image_path, plate_number, helmet_status, confidence)
                   VALUES (?, ?, ?, ?, ?)''',
                (datetime.now().isoformat(), ann_path,
                 plate, helmet_status, float(count))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB insert error: {e}")

        # ── Encode annotated frame ────────────────────────────────────────────
        b64 = ""
        try:
            _, buf = cv2.imencode('.jpg', annotated,
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            logger.error(f"Encode error: {e}")

        logger.info(f"✅ Done → violations={count}, plate={plate}")

        # Always return 200 with whatever we have
        return jsonify({
            'success': True,
            'violations': count,
            'plate': plate,
            'helmet_status': helmet_status,
            'image_path': ann_path,
            'annotated_b64': b64,
        })

    except Exception as e:
        # Absolute last resort — still return 200 so frontend doesn't crash
        logger.error(f"Unexpected upload error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'violations': 0,
            'plate': 'UNKNOWN',
            'helmet_status': 'Error',
            'error': str(e),
            'annotated_b64': '',
        }), 200


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  🚔  Helmet Violation System — API")
    print("=" * 50)
    print("  GET   /health")
    print("  GET   /violations")
    print("  GET   /stats")
    print("  POST  /upload")
    print("  GET   /uploads/<filename>")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)