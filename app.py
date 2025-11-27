from flask import Flask, render_template, request, Response, jsonify
import os, time, base64, threading
import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
os.makedirs("alerts", exist_ok=True)
os.makedirs("uploads", exist_ok=True)  # Folder to store uploaded images/videos

# Load YOLO Models
general_model = YOLO("yolov8n.pt")
weapon_model = YOLO("models/weapon_detector.pt")

# -----------------------------
# Global Variables
# -----------------------------
sender_email = ""
receiver_email = ""
sender_password = ""
detection_targets = []
current_alert = ""
current_target_text = "Detecting all objects."
session_email_sent = False

# VIDEO FILE
video_running = False
video_path = ""

# BROWSER WEBCAM
webcam_running = False

# CCTV / RTSP
cctv_running = False
rtsp_link = ""
latest_cctv_frame = None
cctv_lock = threading.Lock()

# -----------------------------
# EMAIL FUNCTIONS
# -----------------------------
@app.route("/save_emails", methods=["POST"])
def save_emails():
    global sender_email, receiver_email, sender_password
    sender_email = request.form.get("sender_email")
    receiver_email = request.form.get("receiver_email")
    sender_password = request.form.get("sender_password")
    return render_template("home.html", email_message="Email Settings Saved!")

def send_email(image_path, label):
    global session_email_sent
    if session_email_sent or not sender_email or not receiver_email or not sender_password:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = f"⚠ ALERT: {label.upper()} detected!"
        msg.attach(MIMEText(f"A {label} was detected.\nSee attached clean frame.", "plain"))

        with open(image_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        session_email_sent = True
        print("EMAIL SENT SUCCESSFULLY")
    except Exception as e:
        print("Email Error:", e)

# -----------------------------
# TARGETS
# -----------------------------
@app.route("/set_target", methods=["POST"])
def set_target():
    global detection_targets, current_target_text, session_email_sent
    targets = request.form.get("targets", "").lower().strip()
    session_email_sent = False
    if targets:
        detection_targets = [t.strip() for t in targets.split(",")]
        current_target_text = "Detecting only: " + ", ".join(detection_targets)
    else:
        detection_targets = []
        current_target_text = "Detecting all objects."
    return jsonify({"message": current_target_text})

# -----------------------------
# DETECTION FUNCTION
# -----------------------------
def detect_objects(frame):
    global current_alert, session_email_sent

    clean_frame = frame.copy()

    results = general_model(frame)[0]
    weapon_results = weapon_model(frame)[0]

    detections = []

    # Collect general detections
    for box in results.boxes:
        cls = int(box.cls[0])
        label = general_model.names[cls].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((label, x1, y1, x2, y2))

    # Collect weapon detections
    for box in weapon_results.boxes:
        cls = int(box.cls[0])
        label = weapon_model.names[cls].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((label, x1, y1, x2, y2))

    found_target = False  # Track if anything matched target

    # Process detections
    for label, x1, y1, x2, y2 in detections:
        is_target = (not detection_targets) or (label in detection_targets)

        if is_target:
            found_target = True

            color = (0, 0, 255) if label in ["gun", "knife", "weapon"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Set alert
            current_alert = f"⚠ ALERT: {label.upper()} detected!"

            # Email only once
            if not session_email_sent:
                alert_path = f"alerts/{int(time.time())}.jpg"
                cv2.imwrite(alert_path, clean_frame)
                send_email(alert_path, label)

    # Clear alert if nothing found
    if not found_target:
        current_alert = ""

    return frame

# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -----------------------------
# IMAGE DETECTION
# -----------------------------
@app.route("/image", methods=["GET", "POST"])
def image_page():
    global session_email_sent
    session_email_sent = False
    if request.method == "POST":
        file = request.files["file"]
        # Save to uploads folder with timestamp to avoid overwriting
        filename = f"{int(time.time())}_{file.filename}"
        saved_path = os.path.join("uploads", filename)
        file.save(saved_path)

        img = cv2.imread(saved_path)
        output = detect_objects(img)
        _, buffer = cv2.imencode(".jpg", output)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        return render_template("image.html", result_image=img_b64,
                               alert_message=current_alert, target_message=current_target_text)
    return render_template("image.html", result_image=None, alert_message=None,
                           target_message=current_target_text)

# -----------------------------
# VIDEO FILE DETECTION
# -----------------------------
@app.route("/video_page")
def video_page():
    return render_template("video.html", target_message=current_target_text)

@app.route("/video", methods=["POST"])
def video_upload():
    global video_running, video_path, session_email_sent
    session_email_sent = False
    file = request.files["file"]
    # Save to uploads folder
    filename = f"{int(time.time())}_{file.filename}"
    video_path = os.path.join("uploads", filename)
    file.save(video_path)
    video_running = True
    return "OK"

@app.route("/video_feed")
def video_feed():
    def generate_video_frames():
        global video_running
        cap = cv2.VideoCapture(video_path)
        while video_running:
            success, frame = cap.read()
            if not success:
                break
            frame = detect_objects(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        cap.release()
    return Response(generate_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop_video")
def stop_video():
    global video_running
    video_running = False
    return "Video stopped"

# -----------------------------
# BROWSER WEBCAM DETECTION
# -----------------------------
@app.route("/webcam_page")
def webcam_page():
    return render_template("webcam.html", target_message=current_target_text)

@app.route("/webcam_feed", methods=["POST"])
def webcam_feed():
    global session_email_sent
    session_email_sent = False

    data_url = request.form.get("frame")
    if not data_url:
        return jsonify({"error": "No frame sent"}), 400

    try:
        # Decode base64
        header, encoded = data_url.split(",", 1)
        frame_data = base64.b64decode(encoded)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Frame decode failed"}), 400

        # Run detection
        processed_frame = detect_objects(frame)

        # Encode back to JPEG
        _, buffer = cv2.imencode(".jpg", processed_frame)
        processed_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"processed_frame": processed_b64, "alert": current_alert})

    except Exception as e:
        print("Webcam processing error:", e)
        return jsonify({"error": str(e)}), 500

# -----------------------------
# CCTV / RTSP DETECTION
# -----------------------------
@app.route("/cctv_page")
def cctv_page():
    return render_template("cctv.html", target_message=current_target_text)

def cctv_capture_thread():
    global latest_cctv_frame, cctv_running, rtsp_link
    cap = cv2.VideoCapture(rtsp_link)
    frame_skip = 2
    count = 0
    while cctv_running:
        success, frame = cap.read()
        if not success or frame is None:
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_link)
            continue
        count += 1
        if count % frame_skip != 0:
            with cctv_lock:
                latest_cctv_frame = frame
            continue
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        detected_frame = detect_objects(small_frame)
        detected_frame = cv2.resize(detected_frame, (w, h))
        with cctv_lock:
            latest_cctv_frame = detected_frame
    cap.release()

@app.route("/start_rtsp", methods=["POST"])
def start_rtsp():
    global rtsp_link, cctv_running, session_email_sent
    session_email_sent = False
    rtsp_link = request.form.get("rtsp")
    if not rtsp_link:
        return "RTSP link missing", 400
    if not cctv_running:
        cctv_running = True
        threading.Thread(target=cctv_capture_thread, daemon=True).start()
    return "OK"

@app.route("/rtsp_feed")
def rtsp_feed():
    def generate_frames():
        global latest_cctv_frame, cctv_running
        while cctv_running:
            with cctv_lock:
                frame = latest_cctv_frame.copy() if latest_cctv_frame is not None else None
            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            else:
                time.sleep(0.01)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stop_cctv")
def stop_cctv():
    global cctv_running
    cctv_running = False
    return "CCTV stopped"

# -----------------------------
# ALERT FETCH API
# -----------------------------
@app.route("/get_alert")
def get_alert():
    return jsonify({"alert": current_alert})

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)


