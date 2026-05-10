# app.py  ← THE ONLY FLASK FILE YOU NEED
# =========================================
# REST API endpoints:
#   GET  /health
#   GET  /violations
#   GET  /stats
#   POST /upload
#   GET  /uploads/<filename>
# =========================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import cv2
import base64
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return ('.' in filename
            and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('violations.db')
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
# Load detection modules  (detect_production.py + plate_reader.py)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from detect_production import get_detector
    from plate_reader import get_plate_reader

    detector     = get_detector()
    plate_reader = get_plate_reader()
    logger.info("✅ Detection modules loaded")

except Exception as e:
    logger.error(f"Could not load detection modules: {e}")

    # ── Fallback stubs — API stays alive even without models ─────────────────
    class _DummyDetector:
        def detect_violations(self, frame, return_visualization=False):
            logger.warning("⚠️  DummyDetector — train the CNN model first!")
            return ([], frame) if return_visualization else []

    class _DummyReader:
        def extract_plate_text(self, image):
            return "UNKNOWN"

    detector     = _DummyDetector()
    plate_reader = _DummyReader()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "time": datetime.now().isoformat()})


@app.route('/violations', methods=['GET'])
def get_violations():
    try:
        conn = sqlite3.connect('violations.db')
        rows = conn.execute(
            '''SELECT id, timestamp, image_path, plate_number
               FROM violations
               ORDER BY id DESC LIMIT 100'''
        ).fetchall()
        conn.close()
        return jsonify([
            {'id': r[0], 'time': r[1], 'image': r[2], 'plate': r[3] or 'UNKNOWN'}
            for r in rows
        ])
    except Exception as e:
        logger.error(f"DB error: {e}")
        return jsonify([])


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect('violations.db')
        total  = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        weekly = conn.execute(
            "SELECT COUNT(*) FROM violations "
            "WHERE timestamp >= datetime('now', '-7 days')"
        ).fetchone()[0]
        plates = conn.execute(
            "SELECT COUNT(*) FROM violations "
            "WHERE plate_number NOT IN ('UNKNOWN', 'ERROR', '')"
        ).fetchone()[0]
        conn.close()
        return jsonify({'total': total, 'weekly': weekly, 'platesDetected': plates})
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'total': 0, 'weekly': 0, 'platesDetected': 0})


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Accepts an image file (or live-stream JPEG frame).
    Runs helmet detection + plate OCR.
    Saves result to DB and returns JSON with annotated frame.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image in request'}), 400

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'error': 'Empty file'}), 400

        # Accept any image extension (live frames come as 'frame.jpg')
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
        try:
            violations, annotated = detector.detect_violations(
                frame, return_visualization=True)
            count         = len(violations)
            helmet_status = "No Helmet" if count > 0 else "Helmet OK"

            # Save annotated version
            ann_name = f"ann_{safe_name}"
            ann_path = os.path.join(UPLOAD_FOLDER, ann_name)
            cv2.imwrite(ann_path, annotated)

        except Exception as e:
            logger.error(f"Detection error: {e}")
            count, helmet_status, ann_path, annotated = 0, "Error", filepath, frame

        # ── Plate OCR (only when violation exists) ────────────────────────────
        plate = "UNKNOWN"
        if count > 0:
            try:
                plate = plate_reader.extract_plate_text(frame)
            except Exception as e:
                logger.error(f"Plate OCR error: {e}")

        # ── Save to database ──────────────────────────────────────────────────
        conn = sqlite3.connect('violations.db')
        conn.execute(
            '''INSERT INTO violations
               (timestamp, image_path, plate_number, helmet_status, confidence)
               VALUES (?, ?, ?, ?, ?)''',
            (datetime.now().isoformat(), ann_path, plate, helmet_status, float(count))
        )
        conn.commit()
        conn.close()

        # ── Encode annotated frame for React frontend ─────────────────────────
        _, buf = cv2.imencode('.jpg', annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode('utf-8')

        logger.info(f"Result → violations: {count}, plate: {plate}")

        return jsonify({
            'success':       True,
            'violations':    count,
            'plate':         plate,
            'helmet_status': helmet_status,
            'image_path':    ann_path,
            'annotated_b64': b64,          # ready for <img src="data:image/jpeg;base64,...">
        })

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
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