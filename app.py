import os
import cv2
import time
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, render_template, send_file
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "verification_logs.db"
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
LOG_INTERVAL_SECONDS = float(os.getenv("LOG_INTERVAL_SECONDS", "5"))

app = Flask(__name__)

class VerificationCamera:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.status = {
            "camera": "starting",
            "human_detected": False,
            "human_count": 0,
            "item_detected": False,
            "item_count": 0,
            "items": [],
            "overall": "WAITING",
            "message": "Starting camera and AI model..."
        }
        self.running = True
        self.last_log_time = 0.0
        self.last_logged_state = None
        self.cap = None
        self.model = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def update_status(self, **kwargs):
        with self.lock:
            self.status.update(kwargs)

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def log_if_needed(self, status):
        now = time.time()
        state = (
            status["human_detected"],
            status["human_count"],
            status["item_detected"],
            tuple(sorted(status["items"])),
            status["overall"],
        )
        # Record a state change immediately, then at most one repeat every interval.
        if state != self.last_logged_state or now - self.last_log_time >= LOG_INTERVAL_SECONDS:
            save_event(status)
            self.last_logged_state = state
            self.last_log_time = now

    def run(self):
        try:
            self.update_status(camera="loading", message=f"Loading AI model: {MODEL_NAME}")
            self.model = YOLO(MODEL_NAME)
        except Exception as exc:
            self.update_status(camera="error", overall="ERROR",
                               message=f"Could not load model: {exc}")
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.update_status(camera="error", overall="ERROR",
                               message=f"Cannot open camera index {CAMERA_INDEX}. Try CAMERA_INDEX=1.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.update_status(camera="online", message="Camera is running.")

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.update_status(camera="error", overall="ERROR",
                                   message="Camera stopped returning frames.")
                time.sleep(0.2)
                continue

            try:
                result = self.model.predict(frame, conf=CONFIDENCE, imgsz=640, verbose=False)[0]
                human_count = 0
                detected_items = []
                annotated = frame.copy()

                names = result.names
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        label = str(names[class_id])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if label.lower() == "person":
                            human_count += 1
                            color = (40, 200, 40)
                            display_label = f"PERSON {confidence:.0%}"
                        else:
                            detected_items.append(label)
                            color = (255, 160, 30)
                            display_label = f"{label.upper()} {confidence:.0%}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, display_label, (x1, max(22, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

                item_names = sorted(set(detected_items))
                human_detected = human_count > 0
                item_detected = len(detected_items) > 0
                overall = "VERIFIED" if (human_detected or item_detected) else "NOT VERIFIED"

                if human_detected and item_detected:
                    message = "Human and item(s) detected."
                elif human_detected:
                    message = "Human detected. No supported item detected."
                elif item_detected:
                    message = "Item(s) detected. No human detected."
                else:
                    message = "No human or supported item detected."

                status = {
                    "camera": "online",
                    "human_detected": human_detected,
                    "human_count": human_count,
                    "item_detected": item_detected,
                    "item_count": len(detected_items),
                    "items": item_names,
                    "overall": overall,
                    "message": message,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")
                }

                cv2.putText(annotated, f"OVERALL: {overall}", (15, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (40, 200, 40) if overall == "VERIFIED" else (30, 30, 230), 2)

                self.update_status(**status)
                self.log_if_needed(status)

                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with self.lock:
                        self.frame = encoded.tobytes()

            except Exception as exc:
                self.update_status(message=f"Detection error: {exc}")
                time.sleep(0.1)

            time.sleep(0.02)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()


def db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                human_detected INTEGER NOT NULL,
                human_count INTEGER NOT NULL,
                item_detected INTEGER NOT NULL,
                item_count INTEGER NOT NULL,
                items TEXT NOT NULL,
                overall TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)
        conn.commit()


def save_event(status):
    with db_connection() as conn:
        conn.execute("""
            INSERT INTO events
            (timestamp, human_detected, human_count, item_detected, item_count, items, overall, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            int(status["human_detected"]), status["human_count"],
            int(status["item_detected"]), status["item_count"],
            ", ".join(status["items"]), status["overall"], status["message"]
        ))
        conn.commit()


initialize_database()
camera = VerificationCamera()



def generate_mjpeg():
    while True:
        try:
            # Get the latest JPEG frame from the AI camera
            jpg_bytes = camera.get_frame()

            if jpg_bytes is None:
                import numpy as np

                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)

                cv2.putText(
                    placeholder,
                    "Starting AI Camera...",
                    (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                success, encoded = cv2.imencode(".jpg", placeholder)

                if not success:
                    time.sleep(0.1)
                    continue

                jpg_bytes = encoded.tobytes()

            # Send the JPEG frame to the browser
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg_bytes)).encode() +
                b"\r\n\r\n" +
                jpg_bytes +
                b"\r\n"
            )

            time.sleep(0.04)

        except GeneratorExit:
            break

        except Exception as error:
            print("VIDEO STREAM ERROR:", error)
            time.sleep(0.2)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"
        }
    )



@app.route("/api/status")
def api_status():
    return jsonify(camera.get_status())


@app.route("/api/events")
def api_events():
    with db_connection() as conn:
        rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 20").fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/export/csv")
def export_csv():
    with db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY id DESC", conn)
    path = BASE_DIR / "verification_report.csv"
    df.to_csv(path, index=False)
    return send_file(path, as_attachment=True, download_name="verification_report.csv")


@app.route("/export/excel")
def export_excel():
    with db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY id DESC", conn)
    path = BASE_DIR / "verification_report.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return send_file(path, as_attachment=True, download_name="verification_report.xlsx")


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)